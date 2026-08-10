STAGE_NAMES={0:"S0 Normal",1:"S1 Route Attraction",2:"S2 Traffic Concentration",3:"S3 Packet Absorption"}
def make_narrative(stage, next_stage, confidence, risk, risk_level, top_features):
    feats=", ".join(f"{x['feature']} ({x['magnitude']:.3f})" for x in top_features[:3]) or "no dominant feature"
    return (
        f"Session is {STAGE_NAMES[int(stage)]}. "
        f"Predicted next stage: {STAGE_NAMES.get(int(next_stage),'termination/unknown')}. "
        f"Confidence={confidence:.1%}, risk={risk:.1f} ({risk_level}). "
        f"Dominant evidence: {feats}."
    )
