from __future__ import annotations

import numpy as np
import pandas as pd


# Distance bands in miles. Delay dynamics differ for short shuttle hops
# versus long-haul transcontinental flights.
DISTANCE_BINS = [0, 250, 500, 750, 1000, 1500, 2000, 6000]
DISTANCE_LABELS = [
    "0-250",
    "250-500",
    "500-750",
    "750-1000",
    "1000-1500",
    "1500-2000",
    "2000+",
]

# A flight at or above this distance is treated as long-haul.
LONG_HAUL_MILES = 1500


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe, static route features.

    These features describe the route itself and are fixed regardless of
    outcome, so they introduce no leakage. Historical delay statistics by
    route are produced separately by the historical-feature engine and are
    intentionally NOT duplicated here.

    Added features:
    - route            origin>dest identifier (e.g. "JFK>MIA")
    - distance_band    binned flight distance (categorical)
    - is_long_haul     1 if distance >= LONG_HAUL_MILES
    - distance_log     log1p of distance (compresses the long tail)
    """

    required = {"origin", "dest", "distance"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result["route"] = (
        result["origin"].astype(str)
        + ">"
        + result["dest"].astype(str)
    )

    result["distance_band"] = pd.cut(
        result["distance"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        right=False,
    )

    result["is_long_haul"] = (
        result["distance"] >= LONG_HAUL_MILES
    ).astype("int8")

    result["distance_log"] = np.log1p(result["distance"])

    return result
