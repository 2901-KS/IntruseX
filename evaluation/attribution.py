import numpy as np
import torch

class AttributionProvider:
    def explain(self, model, x): raise NotImplementedError

class GradientAttribution(AttributionProvider):
    def explain(self, model, x):
        model.eval(); z=x.clone().detach().requires_grad_(True); out=model(z)
        pred=out["stage_logits"].argmax(-1); score=out["stage_logits"][torch.arange(z.size(0),device=z.device),pred].sum()
        model.zero_grad(set_to_none=True); score.backward()
        return {"method":"gradient","feature_time_abs_grad":z.grad.detach().abs().cpu(),"predicted_stage":pred.detach().cpu()}

class OcclusionAttribution(AttributionProvider):
    def __init__(self, baseline=0.0): self.baseline=baseline
    def explain(self, model, x):
        model.eval();
        with torch.no_grad():
            base=model(x)["stage_logits"]; pred=base.argmax(-1); base_score=base.gather(1,pred[:,None]).squeeze(1)
            vals=torch.zeros_like(x)
            for t in range(x.size(1)):
                xp=x.clone(); xp[:,t,:]=self.baseline
                s=model(xp)["stage_logits"].gather(1,pred[:,None]).squeeze(1)
                vals[:,t,:]=(base_score-s).abs()
        return {"method":"occlusion","feature_time_importance":vals.cpu(),"predicted_stage":pred.cpu()}

class EventTimeAttribution(GradientAttribution):
    def explain(self, model, x):
        result=super().explain(model,x); result["method"]="event_time_gradient"; return result

PROVIDERS={"tgn":GradientAttribution,"evolvegcn":OcclusionAttribution,"graph_wavenet":OcclusionAttribution,
           "tft":GradientAttribution,"timesnet":OcclusionAttribution,"patchtst":OcclusionAttribution,
           "neural_cde":GradientAttribution,"deep_ssm":GradientAttribution,"mamba":GradientAttribution,
           "deephit":EventTimeAttribution}

def get_provider(model_name): return PROVIDERS.get(model_name,GradientAttribution)()

def rule_hint(feature_names, x_last):
    vals=x_last.detach().cpu().abs().numpy()
    top=np.argsort(vals)[::-1][:5]
    return [{"feature":feature_names[int(i)],"magnitude":float(vals[int(i)])} for i in top]

def explain(model_name, model, x): return get_provider(model_name).explain(model,x)
