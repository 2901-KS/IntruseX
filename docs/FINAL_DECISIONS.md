# Final Implementation Decisions

1. The original IntruSex-BH architecture remains the high-level design: telemetry -> normalization -> causal history -> stage detection -> session intelligence -> transition/sojourn -> risk -> adaptive threshold -> XAI -> visualization.
2. W1-W12 edge weights are **not implemented**.
3. `path_ratio` is **not implemented**.
4. No edge/path values are inferred, synthesized, approximated, or used as hidden inputs.
5. The packaged CSV's 33 native telemetry columns are the complete runtime feature contract.
6. Box H is implemented as a Prometheus exporter plus a provisioned Grafana dashboard.
7. The packaged TFT checkpoint is a real trained checkpoint produced from the supplied CSV; it is a deployment artifact, not a claim of real Mininet/Ryu validation.
8. The ten model classes remain available for comparative research, but deployment defaults to the packaged TFT checkpoint.
