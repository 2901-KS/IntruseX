# IntruSex-BH — Deployment-Ready Behaviour-Driven Blackhole Intelligence

This package implements the IntruSex-BH detection/intelligence pipeline using the supplied CSV as the runtime schema.

## Explicit architecture decision

**W1-W12 edge weights and `path_ratio` are completely removed from this implementation.**

They are not generated, inferred, approximated, used as model inputs, used in risk scoring, or shown in the dashboard. The project operates on the telemetry actually present in the supplied CSV:

- 30 native H/G/I telemetry features
- 3 global telemetry features: `adv_rate`, `entropy`, `jsd`
- total runtime input dimension: **33**

The original architecture document and the newer ten-model Markdown are included under `docs/` for traceability. This repository follows the user's final decision wherever it conflicts with those earlier documents.

## Pipeline

```text
Supplied CSV / Mininet-Ryu telemetry
        |
        v
Box A1: train-only normalization
        |
        v
Box A2: causal history adapter
        |
        v
10 candidate backbones
        |
        +--> Box B: S0/S1/S2/S3
        |
        +--> Box C: persistent behavioural sessions
        |
        +--> Box D: masked next-stage + sojourn
        |
        v
Box E: contextual risk
        |
        v
Box F: adaptive alert threshold
        |
        v
Box G: attribution + narrative
        |
        v
Box H: Prometheus metrics -> Grafana dashboard
```

The operational stage lifecycle remains:

- S0 — Normal routing
- S1 — Route attraction
- S2 — Traffic concentration
- S3 — Packet absorption

## Supplied dataset

The packaged dataset is `data/raw/intrusex_bh.csv`.

Observed schema:

- 21,271 rows
- 6 complete runs
- 46 original columns
- 33 runtime telemetry features
- S0/S1/S2/S3 labels

The implementation performs run-level splitting before overlapping causal windows and fits the scaler only on training runs.

## Ten model backbones

The repository contains executable PyTorch implementations behind the shared interface:

1. TGN
2. EvolveGCN
3. Graph WaveNet
4. TFT
5. TimesNet
6. PatchTST
7. Neural CDE-style continuous-time recurrent dynamics
8. Deep nonlinear SSM
9. Mamba-family selective state-space model
10. DeepHit

All return the common current-stage, next-stage, sojourn and latent outputs. DeepHit additionally exposes event-time probability mass.

## Baselines and evaluation

The package also contains:

- Random Forest
- FT-Transformer
- GRU
- feature-weighted semi-Markov baseline
- grouped run-level bootstrap CIs
- temperature calibration
- Brier score / log loss
- ordinal stage error
- robustness scenarios
- attribution providers
- ranking/leaderboard utilities

No final-five claim is hard-coded.

## REST API

The default deployment model is the packaged TFT checkpoint.

### Start locally

```bash
python -m pip install -r requirements.txt
python verify.py
pytest -q
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

```text
GET  /health
GET  /ready
GET  /model
GET  /metrics
POST /predict
```

### Prediction request

`POST /predict`

```json
{
  "dpid": "network",
  "trust_score": 0,
  "history": [
    {
      "rate_H": 0.0,
      "advr_H": 0.0,
      "imb_H": 0.0,
      "grad_H": 0.0,
      "corr_H": 0.0,
      "dacc_H": 0.0,
      "qdrm_H": 0.0,
      "qdrv_H": 0.0,
      "skew_H": 0.0,
      "kurt_H": 0.0,
      "rate_G": 0.0,
      "advr_G": 0.0,
      "imb_G": 0.0,
      "grad_G": 0.0,
      "corr_G": 0.0,
      "dacc_G": 0.0,
      "qdrm_G": 0.0,
      "qdrv_G": 0.0,
      "skew_G": 0.0,
      "kurt_G": 0.0,
      "rate_I": 0.0,
      "advr_I": 0.0,
      "imb_I": 0.0,
      "grad_I": 0.0,
      "corr_I": 0.0,
      "dacc_I": 0.0,
      "qdrm_I": 0.0,
      "qdrv_I": 0.0,
      "skew_I": 0.0,
      "kurt_I": 0.0,
      "adv_rate": 0.0,
      "entropy": 3.0,
      "jsd": 0.01
    }
  ]
}
```

The API returns current stage, next stage, probabilities, confidence, estimated remaining polls, risk level, threshold, alert decision, top evidence and a human-readable narrative.

## Prometheus + Grafana

Run the complete monitoring stack:

```bash
docker compose up --build -d
```

Then open:

```text
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```

Grafana credentials:

```text
username: admin
password: intrusex-admin
```

The `IntruSex-BH Live Intelligence` dashboard is provisioned automatically.

Dashboard panels include:

- current S0-S3 stage
- risk score
- confidence
- estimated remaining sojourn
- current-stage probabilities
- next-stage probabilities
- inference latency
- alerts by severity
- inference request rate

## Train a different model for deployment

```bash
python scripts/train_deploy_model.py --model tgn --epochs 5
```

Then change:

```text
INTRUSEX_MODEL=tgn
INTRUSEX_CHECKPOINT=artifacts/tgn.pt
```

in `docker-compose.yml` and rebuild.

## Train all ten

```bash
python run_all.py --data data/raw/intrusex_bh.csv --epochs 5 --history 32 --stride 8
```

For baselines:

```bash
python run_baselines.py --data data/raw/intrusex_bh.csv --epochs 3
```

## Important scientific boundary

The packaged CSV is the current dataset. The system does not claim real Mininet/Ryu deployment validation merely because the API and Grafana stack are deployable.

When a new real controller dataset is available, use the same 33-feature schema, retrain, recalibrate on validation data, and evaluate on untouched grouped runs.

## Files

```text
api/app.py                         FastAPI inference + Prometheus exporter
data/                              schema, causal windows, scaling
models/                            ten backbones
heads/                             shared multitask heads and losses
baselines/                         RF, FT-Transformer, GRU, semi-Markov
intelligence/                      sessions, risk, thresholds, narratives
evaluation/                        metrics, calibration, robustness, attribution
monitoring/                        Prometheus + Grafana provisioning
artifacts/                         trained deployment checkpoint + scaler
data/raw/                          supplied CSV
tests/                             automated verification
docs/                              supplied Markdown + original architecture reference
docker-compose.yml                 complete monitoring deployment
Dockerfile                         API container
