import torch
from models.models import MODEL_REGISTRY
from evaluation.attribution import rule_hint


def test_all_backbones_forward():
    x=torch.randn(2,16,33)
    stage=torch.tensor([0,1])
    for name,Cls in MODEL_REGISTRY.items():
        model=Cls(33,32)
        out=model(x,stage_target=stage)
        assert out["stage_logits"].shape==(2,4), name
        assert out["next_stage_logits"].shape==(2,4), name
        assert out["sojourn_params"].shape==(2,2), name


def test_rule_hint_import_and_output():
    out=rule_hint(["a","b","c"],torch.tensor([1.,3.,2.]))
    assert out[0]["feature"]=="b"
