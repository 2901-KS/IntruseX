import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from .schema import runtime_columns, inspect_schema, validate_no_removed_columns

STAGES = ["S0", "S1", "S2", "S3"]

@dataclass
class Split:
    train_runs: list
    val_runs: list
    test_runs: list

class CausalScaler:
    """Training-only scaler. Node and global telemetry are transformed separately."""
    def __init__(self):
        self.node_scaler = StandardScaler()
        self.global_scaler = StandardScaler()
        self.node_columns = []
        self.global_columns = []
        self.columns = []

    def fit(self, df):
        validate_no_removed_columns(df)
        schema = inspect_schema(df)
        self.node_columns = schema.node_features
        self.global_columns = schema.global_features
        self.columns = schema.runtime_columns
        if self.node_columns:
            self.node_scaler.fit(
                df[self.node_columns].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
            )
        if self.global_columns:
            self.global_scaler.fit(
                df[self.global_columns].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
            )
        return self

    def transform(self, df):
        parts=[]
        if self.node_columns:
            x=self.node_scaler.transform(
                df[self.node_columns].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
            )
            parts.append(x)
        if self.global_columns:
            x=self.global_scaler.transform(
                df[self.global_columns].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
            )
            parts.append(x)
        return np.concatenate(parts,axis=1).astype("float32")

def grouped_split(df, seed=42):
    runs = sorted(df["run_id"].dropna().unique().tolist())
    if len(runs) < 3:
        raise ValueError("Need at least 3 complete runs.")
    rng=np.random.default_rng(seed)
    rng.shuffle(runs)
    n=len(runs)
    n_test=max(1, round(n*0.17))
    n_val=max(1, round(n*0.17))
    while n_test+n_val>=n:
        if n_test>1: n_test-=1
        elif n_val>1: n_val-=1
        else: break
    return Split(sorted(runs[n_test+n_val:]), sorted(runs[n_test:n_test+n_val]), sorted(runs[:n_test]))

def add_targets(df):
    df=df.sort_values(["run_id","timestamp"]).copy()
    df["next_stage_target"]=-1
    df["sojourn_target"]=0.0
    df["event_observed"]=0
    for run_id,g in df.groupby("run_id",sort=False):
        idx=g.index.to_numpy()
        stages=g["stage"].to_numpy(dtype=int)
        for k,ix in enumerate(idx):
            cur=stages[k]
            j=k+1
            while j<len(stages) and stages[j]==cur:
                j+=1
            if j<len(stages):
                df.loc[ix,"next_stage_target"]=int(stages[j])
                df.loc[ix,"sojourn_target"]=float(j-k)
                df.loc[ix,"event_observed"]=1
    return df

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, history=32, stride=4, horizon=64):
        self.df=df.sort_values(["run_id","timestamp"]).copy()
        self.x=scaler.transform(self.df)
        self.history=history
        self.horizon=horizon
        self.samples=[]
        for run_id,g in self.df.groupby("run_id",sort=False):
            pos=np.flatnonzero(self.df.index.isin(g.index))
            if len(pos)<history: continue
            for end in range(history-1,len(pos),stride):
                self.samples.append((pos[end-history+1:end+1],pos[end]))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        wp,absolute=self.samples[i]
        row=self.df.iloc[absolute]
        return {
            "x":torch.from_numpy(self.x[wp]),
            "stage":torch.tensor(int(row.stage),dtype=torch.long),
            "next_stage":torch.tensor(int(row.next_stage_target),dtype=torch.long),
            "event_observed":torch.tensor(float(row.event_observed),dtype=torch.float32),
            "sojourn":torch.tensor(min(float(row.sojourn_target),self.horizon),dtype=torch.float32),
            "run_id":str(row.run_id),
            "timestamp":str(row.timestamp),
        }

def make_datasets(df, history=32, stride=4, seed=42):
    validate_no_removed_columns(df)
    df=add_targets(df)
    split=grouped_split(df,seed)
    train=df[df.run_id.isin(split.train_runs)].copy()
    scaler=CausalScaler().fit(train)
    datasets={}
    for name,runs in [("train",split.train_runs),("val",split.val_runs),("test",split.test_runs)]:
        datasets[name]=WindowDataset(df[df.run_id.isin(runs)].copy(),scaler,history,stride)
    return datasets,scaler,split
