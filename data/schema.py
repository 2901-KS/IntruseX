from dataclasses import dataclass
from typing import List
import pandas as pd

# This deployment intentionally uses ONLY the telemetry available in the supplied CSV.
# W1-W12 and path_ratio are excluded from the project entirely.
META_COLUMNS = {
    "timestamp", "t_rel", "run_id", "episode_id", "window_pure", "t_in_stage",
    "stage", "stage_name", "label", "gt_dropped", "next_stage_target",
    "sojourn_target", "event_observed", "deactivate_event", "attack_activation_id",
}
NODE_SUFFIXES = ("H", "G", "I")
NODE_PREFIXES = ("rate", "advr", "imb", "grad", "corr", "dacc", "qdrm", "qdrv", "skew", "kurt")
GLOBAL_FEATURES = ("adv_rate", "entropy", "jsd")

@dataclass(frozen=True)
class DatasetSchema:
    node_features: List[str]
    global_features: List[str]

    @property
    def runtime_columns(self) -> List[str]:
        return self.node_features + self.global_features

    @property
    def dimension(self) -> int:
        return len(self.runtime_columns)

def inspect_schema(df: pd.DataFrame) -> DatasetSchema:
    node_features = [
        f"{p}_{s}" for s in NODE_SUFFIXES for p in NODE_PREFIXES
        if f"{p}_{s}" in df.columns
    ]
    global_features = [c for c in GLOBAL_FEATURES if c in df.columns]
    if not node_features:
        raise ValueError("No native H/G/I telemetry feature columns were found.")
    return DatasetSchema(node_features, global_features)

def runtime_columns(df: pd.DataFrame) -> List[str]:
    return inspect_schema(df).runtime_columns

def leakage_columns(df: pd.DataFrame) -> List[str]:
    return sorted(c for c in META_COLUMNS if c in df.columns)

def validate_no_removed_columns(df: pd.DataFrame) -> None:
    forbidden = [c for c in ("W1","W2","W3","W4","W5","W6","W7","W8","W9","W10","W11","W12","path_ratio")
                 if c in df.columns]
    # The supplied CSV should not contain these. If a future file does, fail loudly
    # instead of silently using them.
    if forbidden:
        raise ValueError(
            "This deployment is configured without edge weights/path ratios. "
            f"Unexpected removed columns found: {forbidden}"
        )
