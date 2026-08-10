
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class ModelOutput:
    stage_logits: torch.Tensor
    stage_probabilities: torch.Tensor
    predicted_stage: torch.Tensor
    next_stage_logits: torch.Tensor
    next_stage_probabilities: torch.Tensor
    sojourn_distribution: Any
    estimated_sojourn_remaining: torch.Tensor
    confidence: torch.Tensor
    latent_embedding: torch.Tensor
    attribution_payload: Any

class ModelAdapter:
    """Common boundary required by the ten-model implementation."""
    def __init__(self, backbone):
        self.backbone=backbone
    def __call__(self,x):
        from heads.multitask import decode_output
        return decode_output(self.backbone(x))
