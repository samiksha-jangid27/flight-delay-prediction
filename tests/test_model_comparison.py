"""Synthetic checks for the model-only integration fixes (no held-out test)."""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from flight_delay.features.model_features import (
    FEATURE_SETS, select_model_features,
)
from flight_delay.models.logistic import make_logistic_pipeline, run_experiment
from flight_delay.models.xgboost_model import select_xgboost_features


@pytest.fixture
def synthetic_flights():
    rng = np.random.default_rng(55)
    n = 120
    times = pd.Series(pd.date_range('2013-01-01',periods=20).repeat(6))
    hour = np.arange(n) % 23
    frame = pd.DataFrame({
        'year': times.dt.year,
        'month': times.dt.month,
        'day': times.dt.day,
        'hour': hour,
        'minute': np.repeat(15,n),
        'day_of_week': times.dt.dayofweek,
        'is_weekend': (times.dt.dayofweek >= 5).astype(int),
        'season': np.repeat(1,n),
        'sched_dep_time': hour*100+15,
        'sched_arr_time': (hour*100+315)%2400,
        'distance': np.resize([100,1000,1800,450],n),
        'carrier': np.resize(['AA','UA','DL','B6'],n),
        'origin': np.resize(['JFK','LGA','EWR'],n),
        'dest': np.resize(['MIA','BOS','ORD','LAX'],n),
    })
    for col,p in [('hour',24),('month',12),('day_of_week',7)]:
        frame[f'{col}_sin']=np.sin(2*np.pi*frame[col]/p)
        frame[f'{col}_cos']=np.cos(2*np.pi*frame[col]/p)
    frame['route']=frame['origin']+'>'+frame['dest']
    frame['distance_band']=pd.cut(frame['distance'],bins=[0,250,1500,2500],labels=['0-250','250-1500','1500+'])
    frame['is_long_haul']=(frame['distance']>=1500).astype(int)
    frame['distance_log']=np.log1p(frame['distance'])
    for entity in ('carrier','origin','destination','route'):
        frame[f'{entity}_historical_delay_rate']=rng.random(n)
        frame[f'{entity}_historical_flight_count']=rng.integers(0,100,n)
    frame['origin_hourly_departures']=rng.integers(1,40,n)
    frame['dest_hourly_arrivals']=rng.integers(1,10,n)
    frame['is_delayed']=(np.arange(n)%5<2).astype(int)
    frame['arr_delay']=frame['is_delayed']*33
    frame['dep_delay']=frame['is_delayed']*35
    frame['id']=np.arange(n)
    return frame.iloc[:90].copy(),frame.iloc[90:].copy()


def test_approved_features_match_legacy_selector(synthetic_flights):
    train,_ = synthetic_flights
    assert len(FEATURE_SETS['full']) == 33
    pd.testing.assert_frame_equal(
        select_model_features(train),select_xgboost_features(train)
    )
    assert 'arr_delay' not in select_model_features(train).columns
    assert 'dep_delay' not in select_model_features(train).columns


def test_logistic_imports_no_xgboost_module():
    import flight_delay.models.logistic as logistic
    source=inspect.getsource(logistic)
    assert 'flight_delay.models.xgboost_model' not in source


def test_logistic_invalid_params_rejected():
    with pytest.raises(ValueError,match='positive'):
        make_logistic_pipeline(C=-1)
    with pytest.raises(ValueError,match='class_weight'):
        make_logistic_pipeline(class_weight='foo')


def test_logistic_runner_loads_once_and_saves_model(synthetic_flights,monkeypatch,tmp_path):
    import flight_delay.models.logistic as logistic
    train,val=synthetic_flights
    calls=[]
    def fake_build(path):
        calls.append(path)
        return train,val,object()
    monkeypatch.setattr(logistic,'build_data_pipeline',fake_build)
    result=run_experiment('synthetic.csv',output_dir=tmp_path/'results',model_dir=tmp_path/'models',class_weight='balanced')
    assert len(calls)==1
    assert result['output_path'].exists()
    assert result['model_path'].exists()
    saved=pd.read_csv(result['output_path'])
    assert len(saved)==len(val)
    assert saved['flight_id'].tolist()==val['id'].tolist()
    assert saved['y_true'].tolist()==val['is_delayed'].tolist()
    restored=joblib.load(result['model_path'])
    prob=restored.predict_proba(select_model_features(val))[:,1]
    np.testing.assert_allclose(prob,saved['y_probability'].to_numpy())


def test_both_models_use_same_validation_rows(synthetic_flights):
    from scripts.compare_models import compare_model_candidates
    train,val=synthetic_flights
    results=compare_model_candidates(train,val,n_jobs=1,xgb_params={
        'n_estimators':15,'max_depth':2,'min_child_weight':1,
    })
    assert set(results['model'])=={'logistic_unweighted','logistic_balanced','xgboost_tuned_full'}
    assert results['validation_flights'].eq(len(val)).all()
    assert results['validation_prevalence'].eq(val['is_delayed'].mean()).all()
    assert results['average_precision'].between(0,1).all()
    assert results['roc_auc'].between(0,1).all()


def test_reject_overlapping_dates(synthetic_flights):
    from scripts.compare_models import compare_model_candidates
    train,val=synthetic_flights
    overlap=val.copy()
    overlap[['year','month','day']]=train.iloc[-1][['year','month','day']].to_numpy()
    with pytest.raises(ValueError,match='precede'):
        compare_model_candidates(train,overlap,n_jobs=1)
