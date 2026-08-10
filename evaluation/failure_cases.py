import numpy as np
import pandas as pd

def fixed_attack_timing(df, seed=42):
    out=df.copy(); rng=np.random.default_rng(seed)
    for _,g in out.groupby("run_id"):
        vals=g["t_rel"].to_numpy() if "t_rel" in g else np.arange(len(g))
        out.loc[g.index,"t_rel_shifted"]=np.roll(vals,rng.integers(0,len(vals)))
    return out

def benign_spike(df,magnitude=1.5):
    out=df.copy()
    for c in out.columns:
        if c.startswith(("grad_","rate_")): out[c]=out[c]*magnitude
    return out

def selective_drop_proxy(df):
    out=df.copy()
    for c in out.columns:
        if c.startswith(("qdrm_","qdrv_")): out[c]=out[c]*0.25
    return out

def flash_crowd(df):
    out=df.copy()
    for c in out.columns:
        if c.startswith("rate_"): out[c]=out[c]*1.8
    return out

def scenario_suite(df):
    return {
        "fixed_attack_timing":fixed_attack_timing(df),
        "benign_spike":benign_spike(df),
        "selective_drop":selective_drop_proxy(df),
        "flash_crowd":flash_crowd(df),
    }
