import os, tempfile
import numpy as np
import pandas as pd
import torch

from data.schema import inspect_schema, runtime_columns, validate_no_removed_columns
from data.dataset import make_datasets
from models.models import MODEL_REGISTRY
from intelligence.risk import risk_score, AdaptiveThresholdManager
from baselines.semimarkov import FeatureWeightedSemiMarkov

CSV=os.path.join(os.path.dirname(os.path.dirname(__file__)),"data","raw","intrusex_bh.csv")

def test_schema_excludes_removed_columns():
    df=pd.read_csv(CSV)
    validate_no_removed_columns(df)
    assert len(runtime_columns(df))==33
    assert not any(c.startswith("W") for c in runtime_columns(df))
    assert "path_ratio" not in runtime_columns(df)

def test_grouped_causal_dataset():
    df=pd.read_csv(CSV)
    ds,scaler,split=make_datasets(df,history=8,stride=32,seed=42)
    assert set(split.train_runs).isdisjoint(split.val_runs)
    assert set(split.train_runs).isdisjoint(split.test_runs)
    assert set(split.val_runs).isdisjoint(split.test_runs)
    assert len(ds["train"])>0 and len(ds["val"])>0 and len(ds["test"])>0
    assert ds["train"][0]["x"].shape[-1]==33

def test_all_backbones_forward():
    x=torch.randn(2,8,33)
    for name,Cls in MODEL_REGISTRY.items():
        model=Cls(33,24)
        out=model(x)
        assert out["stage_logits"].shape==(2,4), name
        assert out["next_stage_logits"].shape==(2,4), name

def test_semimarkov_and_risk():
    X=np.random.default_rng(1).normal(size=(40,33))
    stage=np.array([0]*10+[1]*10+[2]*10+[3]*10)
    nxt=np.array([1]*10+[2]*10+[3]*10+[-1]*10)
    dur=np.array([5]*40,float)
    sm=FeatureWeightedSemiMarkov().fit(X,stage,nxt,dur)
    pred,p,dd=sm.predict(X,stage)
    assert p.shape==(40,4)
    score,level=risk_score(2,next_stage=3,confidence=.9)
    assert score>0 and level in {"LOW","MEDIUM","HIGH","CRITICAL"}
    mgr=AdaptiveThresholdManager()
    assert 20<=mgr.threshold(0,2,.9)<=200
