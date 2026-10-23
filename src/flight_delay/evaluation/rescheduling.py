"""Offline cost-sensitive rescheduling what-if analysis on calibrated probabilities.

This module never fits, selects or calibrates anything. It replays frozen
pre-departure delay probabilities under an *explicit, stated* operational cost
model and reports the retrospective net impact of a proactive-rescheduling
policy, plus sensitivity across assumed cost ratios. No real intervention cost
has been approved; all monetary quantities are expressed in units of the delay
cost ``cost_delay`` and every number is a planning-stage what-if, not a saving
that would necessarily be realised in operation. See
``docs/rescheduling_analysis.md`` for the assumptions and their limitations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .metrics import validate


@dataclass(frozen=True)
class CostModel:
    """Explicit homogeneous per-flight cost assumptions for one proactive action.

    ``cost_action`` is paid whenever the policy acts on a flight, regardless of
    outcome. ``cost_delay`` is the disruption cost of a substantially delayed
    arrival that was not acted on. ``recovery`` in [0, 1] is the fraction of
    that delay cost avoided when the action is taken on a flight that does
    delay; the residual ``(1 - recovery) * cost_delay`` is still incurred.
    Costs are identical across flights and flights are treated as independent;
    both are simplifications documented as limitations.
    """

    cost_delay: float = 1.0
    cost_action: float = 0.25
    recovery: float = 0.8

    def __post_init__(self) -> None:
        if not (self.cost_delay > 0 and self.cost_action >= 0):
            raise ValueError("Require cost_delay > 0 and cost_action >= 0")
        if not 0.0 <= self.recovery <= 1.0:
            raise ValueError("recovery must lie in [0, 1]")

    @property
    def cost_ratio(self) -> float:
        """Action cost as a fraction of delay cost, ``C_a / C_d``."""
        return self.cost_action / self.cost_delay

    @property
    def benefit_per_true_action(self) -> float:
        """Net cost avoided by acting on one flight that truly delays."""
        return self.recovery * self.cost_delay - self.cost_action

    @property
    def optimal_threshold(self) -> float | None:
        """Cost-optimal probability boundary ``C_a / (recovery * C_d)``.

        Acting on a flight has non-negative expected value iff its calibrated
        delay probability is at least this value. ``None`` when ``recovery`` is
        zero (acting can never help) or the boundary exceeds one (acting is
        never worthwhile for any probability).
        """
        if self.recovery == 0.0:
            return None
        tau = self.cost_action / (self.recovery * self.cost_delay)
        return float(tau) if tau <= 1.0 else None


def _counts(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    """Confusion counts for the decision ``p >= threshold`` (act on positives)."""
    acted = p >= threshold
    delayed = y == 1
    return {
        "acted": int(acted.sum()),
        "tp": int((acted & delayed).sum()),
        "fp": int((acted & ~delayed).sum()),
        "fn": int((~acted & delayed).sum()),
        "tn": int((~acted & ~delayed).sum()),
    }


def policy_cost(tp: float, fp: float, fn: float, cost: CostModel) -> float:
    """Total realised cost of a policy from its acted/delayed confusion counts."""
    residual_on_acted = (1.0 - cost.recovery) * cost.cost_delay * tp
    return cost.cost_action * (tp + fp) + cost.cost_delay * fn + residual_on_acted


def no_action_cost(n_delayed: float, cost: CostModel) -> float:
    """Status-quo cost: every delayed arrival is incurred unmitigated."""
    return cost.cost_delay * n_delayed


def reschedule_all_cost(n: float, n_delayed: float, cost: CostModel) -> float:
    """Cost of acting on every flight, true delay or not."""
    return cost.cost_action * n + (1.0 - cost.recovery) * cost.cost_delay * n_delayed


def oracle_cost(n_delayed: float, cost: CostModel) -> float:
    """Perfect-foresight lower bound: act on a true delay only when it helps."""
    acted_delay = cost.cost_action + (1.0 - cost.recovery) * cost.cost_delay
    return n_delayed * min(cost.cost_delay, acted_delay)


def policy_outcome(
    y: ArrayLike, p: ArrayLike, threshold: float, cost: CostModel
) -> dict:
    """Score one action threshold: counts, costs and savings versus baselines.

    ``net_savings`` is the status-quo cost minus the policy cost and equals the
    closed form ``tp * benefit_per_true_action - fp * cost_action``. The
    oracle-efficiency fraction reports how much of the attainable saving the
    policy captures; it is ``None`` when no positive saving is attainable.
    """
    y, p = validate(y, p)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    c = _counts(y, p, threshold)
    n = len(y)
    n_delayed = int(y.sum())
    baseline = no_action_cost(n_delayed, cost)
    total = policy_cost(c["tp"], c["fp"], c["fn"], cost)
    net = baseline - total
    oracle_saving = baseline - oracle_cost(n_delayed, cost)
    return {
        "threshold": float(threshold),
        **c,
        "n": n,
        "n_delayed": n_delayed,
        "predicted_positive_rate": c["acted"] / n,
        "precision": (c["tp"] / c["acted"]) if c["acted"] else 0.0,
        "recall": (c["tp"] / n_delayed) if n_delayed else 0.0,
        "policy_cost": total,
        "no_action_cost": baseline,
        "reschedule_all_cost": reschedule_all_cost(n, n_delayed, cost),
        "net_savings": net,
        "net_savings_per_flight": net / n,
        "savings_fraction_of_status_quo": (net / baseline) if baseline else 0.0,
        "oracle_savings": oracle_saving,
        "oracle_efficiency": (net / oracle_saving) if oracle_saving > 0 else None,
        "beats_no_action": net > 0,
        "beats_reschedule_all": total < reschedule_all_cost(n, n_delayed, cost),
    }


def net_benefit_curve(
    y: ArrayLike, p: ArrayLike, cost: CostModel, grid: int = 201
) -> pd.DataFrame:
    """Net savings (in ``cost_delay`` units) at every distinct score and a grid.

    Savings are reported as a fraction of the delay cost so the curve does not
    depend on the absolute cost scale, only on ``cost_ratio`` and ``recovery``.
    This is the cost-sensitive net-benefit of acting on ``p >= threshold`` and
    generalises decision-curve net benefit with an explicit recovery factor.
    """
    y, p = validate(y, p)
    ts = np.unique(np.r_[np.linspace(0.0, 1.0, grid), p])
    order = np.argsort(p)
    scores = p[order]
    labels = y[order]
    cumulative = np.r_[0, np.cumsum(labels)]
    cut = np.searchsorted(scores, ts, side="left")
    tp = y.sum() - cumulative[cut]
    acted = len(y) - cut
    fp = acted - tp
    savings = (cost.benefit_per_true_action * tp - cost.cost_action * fp) / cost.cost_delay
    return pd.DataFrame(
        {
            "threshold": ts,
            "acted": acted,
            "tp": tp,
            "fp": fp,
            "net_savings_per_delay_cost": savings,
            "net_savings_per_flight": savings * cost.cost_delay / len(y),
        }
    )


def cost_ratio_sweep(
    y: ArrayLike,
    p: ArrayLike,
    *,
    recovery: float = 0.8,
    cost_delay: float = 1.0,
    ratios: ArrayLike = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
) -> pd.DataFrame:
    """Policy value at the cost-optimal threshold for a range of cost ratios.

    For each assumed ``cost_ratio = cost_action / cost_delay`` the cost-optimal
    boundary is ``ratio / recovery``; the row records the resulting policy
    outcome. Rows whose optimal boundary exceeds one mean acting is never
    worthwhile at that cost ratio and are reported with an empty policy.
    """
    y, p = validate(y, p)
    rows = []
    for ratio in np.atleast_1d(np.asarray(ratios, dtype=float)):
        cost = CostModel(
            cost_delay=cost_delay,
            cost_action=float(ratio) * cost_delay,
            recovery=recovery,
        )
        tau = cost.optimal_threshold
        outcome = policy_outcome(y, p, 1.0 if tau is None else tau, cost)
        rows.append(
            {
                "cost_ratio": float(ratio),
                "recovery": recovery,
                "optimal_threshold": tau,
                "acted": outcome["acted"],
                "tp": outcome["tp"],
                "fp": outcome["fp"],
                "precision": outcome["precision"],
                "recall": outcome["recall"],
                "net_savings_per_flight": outcome["net_savings_per_flight"],
                "savings_fraction_of_status_quo": outcome[
                    "savings_fraction_of_status_quo"
                ],
                "oracle_efficiency": outcome["oracle_efficiency"],
                "beats_no_action": outcome["beats_no_action"],
                "beats_reschedule_all": outcome["beats_reschedule_all"],
            }
        )
    return pd.DataFrame(rows)


def rescheduling_analysis(
    frame: pd.DataFrame,
    *,
    cost: CostModel | None = None,
    frozen_threshold: float | None = None,
    ratios: ArrayLike = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
) -> dict:
    """Assemble the offline what-if report for one calibrated prediction frame.

    ``frame`` must carry ``y_true`` and ``y_probability`` (the saved-prediction
    schema). ``frozen_threshold`` is the model's operating threshold from
    validation selection, reported as one extra operating point; it is a
    classification threshold, not a cost-derived one, and is labelled as such.
    """
    required = {"y_true", "y_probability"}
    if not required <= set(frame) or frame.empty:
        raise ValueError("frame must provide y_true and y_probability")
    cost = cost or CostModel()
    y = frame.y_true.to_numpy()
    p = frame.y_probability.to_numpy()
    validate(y, p)
    report: dict = {
        "assumptions": {
            "cost_delay": cost.cost_delay,
            "cost_action": cost.cost_action,
            "cost_ratio": cost.cost_ratio,
            "recovery": cost.recovery,
            "optimal_threshold": cost.optimal_threshold,
            "units": "all costs expressed relative to cost_delay",
            "model": (
                "per-flight homogeneous costs; flights independent; offline "
                "retrospective replay with no feedback effects; no operational "
                "cost has been approved"
            ),
        },
        "population": {
            "n": int(len(y)),
            "n_delayed": int(y.sum()),
            "prevalence": float(y.mean()),
        },
        "baselines_per_flight": {
            "no_action": no_action_cost(int(y.sum()), cost) / len(y),
            "reschedule_all": reschedule_all_cost(len(y), int(y.sum()), cost)
            / len(y),
            "oracle": oracle_cost(int(y.sum()), cost) / len(y),
        },
        "policy_at_optimal_threshold": (
            policy_outcome(y, p, cost.optimal_threshold, cost)
            if cost.optimal_threshold is not None
            else None
        ),
        "cost_ratio_sweep": cost_ratio_sweep(
            y, p, recovery=cost.recovery, cost_delay=cost.cost_delay, ratios=ratios
        ).to_dict("records"),
    }
    if frozen_threshold is not None:
        report["policy_at_frozen_threshold"] = {
            "note": "validation-selected classification threshold; not cost-derived",
            **policy_outcome(y, p, float(frozen_threshold), cost),
        }
    return report
