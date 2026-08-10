# Implementing Ten Advanced Models in IntruSex-BH SDN

## Executive Summary

- **One Contract, Ten Backbones**: The models expect fundamentally different inputs, from timed graph events in TGN [executive_summary[0]] [9] to channel-independent patches in PatchTST [executive_summary[1]] [15] -> preserve Boxes A-H, but add model-specific adapters that all expose the same stage, transition, sojourn, confidence, and attribution outputs.
- **Graph Models Fit the Attack Mechanism**: TGN models timed graph events, EvolveGCN evolves graph-convolution parameters with an RNN [executive_summary[2]] [5], and Graph WaveNet combines learned dependencies with dilated temporal convolutions [executive_summary[3]] [6] -> these should be the strongest candidates for detecting route attraction and traffic concentration.
- **Prediction Must Be Multi-Task**: Box B and Box D should not be trained as unrelated systems -> each backbone should learn current stage, next stage, and remaining stage duration jointly, while the existing feature-weighted semi-Markov implementation remains a mandatory baseline.
- **DeepHit Is a Predictor, Not a Complete Detector**: DeepHit directly learns event-time distributions and supports competing risks [executive_summary[4]] [2] -> use it primarily for Box D, with an auxiliary current-stage head if it must enter the ten-model end-to-end comparison.
- **Synthetic Results Are Provisional**: A common synthetic-data utility test is to train on synthetic data and test on real data [executive_summary[5]] [16] -> do not declare the final five until every candidate has been evaluated on untouched Mininet/Ryu runs containing real controller measurements.
- **Early Stages Matter More Than Accuracy**: Macro F1 gives every stage equal weight because it averages the class-specific F1 values without support weighting [executive_summary[6]] [13] -> make S1 recall, S2 recall, transition lead time, and false alerts per hour first-class metrics rather than selecting models by accuracy.
- **Provisional First Five**: Before results exist, the best architecture-fit shortlist is TGN, Graph WaveNet, TFT, Mamba, and DeepHit -> treat this only as the first implementation wave, not as the final performance ranking.
- **Complexity Must Earn Its Place**: If a complex candidate cannot beat Random Forest or FT-Transformer for Box B and the existing semi-Markov model for Box D, within the same real-data test and latency budget, it should not enter the final five.

## 1. Preserve Boxes A-H With a Shared Multi-Task Interface

Do not install ten independent pipelines. Build one `ModelAdapter` interface and swap only the input adapter and backbone. Boxes E-H should never need to know whether the active model is TGN, TFT, Mamba, or DeepHit.

The revised flow should be:

```text
Phase 1 telemetry
    -> Box A1: train-fitted normalization
    -> Box A2: causal history and graph adapter
    -> Candidate backbone
    -> Box B head: current S0-S3 stage
    -> Box C: persistent session state
    -> Box D heads: next stage and remaining sojourn
    -> Boxes E-F: risk and alert decision
    -> Box G: model-specific attribution plus rule explanation
    -> Box H: common dashboard schema
```

Box A2 is necessary because the present Box B consumes one row at a time, but all ten proposed models require either a sequence, graph history, or event history. Box A2 should maintain a causal buffer keyed by `run_id` and `dpid`; it must contain only observations at or before the current poll. Box C remains the semantic session builder after classification. In other words, the Box A2 buffer supplies computational context, while Box C supplies attack-session meaning.

Every adapted model should return this object:

```python
ModelOutput = {
    "stage_logits": [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[0]] [4],
    "stage_probabilities": [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[0]] [4],
    "predicted_stage": int,
    "next_stage_logits": [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[0]] [4],
    "next_stage_probabilities": [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[0]] [4],
    "sojourn_distribution": object,
    "estimated_sojourn_remaining": float,
    "confidence": float,
    "latent_embedding": vector,
    "attribution_payload": object
}
```

Apply a transition mask after `next_stage_logits`. Allow normal progression and explicitly recorded deactivation or mitigation paths, but reject impossible jumps. For example, S3 -> S0 is permitted only when the controller recorded an explicit deactivate or recovery event; otherwise Box C should mark it anomalous as your locked design specifies.

This shared interface makes comparison meaningful. TGN has native graph memory [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[1]] [9], TFT is natively designed for interpretable multi-horizon forecasting [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[0]] [4], and DeepHit predicts event type and time [1_preserve_boxes_a_h_with_a_shared_multi_task_interface[2]] [2]. They can still be compared if their native representations feed identical operational outputs.

**Decision-ready insight:** keep the architecture locked at the box boundary, not at the internal model type. Add Box A2 and standardized heads; do not rewrite Boxes E-H ten times.

## 2. Build Three Canonical Data Views From One Telemetry Stream

The current CSV schema is sufficient for tabular baselines but not yet ideal for graph and temporal models. Keep the 10 native node features, 12 edge weights, and `path_ratio`, but persist the identifiers and targets needed to reconstruct causally valid histories.

### Required stored fields

```text
run_id, topology_id, seed, timestamp, poll_index, dpid
attack_activation_id, deactivate_event
S0-S3 ground-truth label
10 node features
W1-W12
path_ratio
```

Offline preprocessing should derive, but never feed as runtime inputs, `session_id`, `next_stage`, `polls_to_transition`, `event_type`, and `event_observed`. Ground-truth stages should come from the simulator/controller attack schedule and state machine, not from thresholds over QDR, entropy, or packet drops. If the same feature rule both creates the label and generates the feature, synthetic test performance will mostly measure recovery of the generator's rule.

| Canonical view | Shape or record | Models | Treatment of the 23 telemetry values |
|---|---|---|---|
| Timed-event graph | `(src, dst, time, message)` | TGN | Convert probe, forwarding, port-stat delta, and edge-weight changes into timestamped events. Attach endpoint features and `path_ratio` to the event message. |
| Graph snapshots | `X: [L, 9, F]`, `A/E: [L, 12, E]` | EvolveGCN, Graph WaveNet | Keep 10 features per switch as node channels. Keep W1-W12 as edge attributes or dynamic supports. Keep `path_ratio` as global context. |
| Multivariate sequence | `[L, C]` per switch or `[L, 103]` network-wide | TFT, TimesNet, PatchTST, Neural CDE, Deep SSM, Mamba | A network snapshot can contain `9 x 10 + 12 + 1 = 103` values. A per-switch sequence may use 23 values, but the 12 edge values and path ratio are then shared global context. |
| Landmark survival record | `(history, event_type, time_to_event, censor)` | DeepHit | At each valid landmark, use the causal history as covariates and the next transition plus remaining polls as targets. |

For graph models, represent the logical s7-s8 route as an explicitly typed edge with fields such as `is_physical`, `is_probe_advertised`, `active`, `weight`, and `delta_weight`. Do not silently add it to the physical adjacency matrix. That distinction is one of the most informative facts in your attack model.

Normalize using training data only. Apply `log1p` or another documented count transform to highly skewed packet and edge counts before scaling. Fit separate transformations for node features, edge features, and global path features as your architecture requires. Neural CDE must additionally receive timestamps, observation masks, and missingness indicators because its defining use case is partially observed, irregularly sampled series [2_build_three_canonical_data_views_from_one_telemetry_stream[0]] [3].

Split by complete simulation run before creating overlapping windows. Research examples explicitly keep samples with identical geometry in the same split to prevent leakage [2_build_three_canonical_data_views_from_one_telemetry_stream[1]] [7]. For IntruSex-BH, the equivalent unit is the complete run, topology, attack activation, and seed, not an individual CSV row.

**Decision-ready insight:** one CSV should generate four data views. Never flatten away edge identity for graph models, and never split overlapping windows from the same run across train and test.

## 3. How to Implement Each of the Ten Models

### 3.1 Temporal Graph Network: the event-native candidate

**Role:** Box B plus Box D. TGN is the closest match when Ryu can preserve every probe, `PACKET_IN`, `FLOW_MOD`, forwarding, and port-stat event with a timestamp. TGN represents dynamic graphs as sequences of timed events and combines memory modules with graph operators [3_how_to_implement_each_of_the_ten_models[0]] [9].

**Input:** Create one event message containing edge-weight delta, packet rate, probe type, endpoint QDR and CC, entropy change, and `path_ratio`. Maintain memory for all nine switches. The current embedding of each switch feeds a four-class node-stage head; attention or mean pooling over all switch memories feeds the network-stage and transition heads.

**Prediction:** Concatenate the current switch memory, global graph embedding, current stage probability, and selected current features. Feed this into masked next-stage logits and a Weibull or discrete-hazard sojourn head. The official TGN repository expects timed-event graph data [3_how_to_implement_each_of_the_ten_models[1]] [23].

**Concrete case:** When s8 advertises the virtual relation to s7, TGN receives a typed probe event before packet absorption begins. Its memory can combine that event with subsequent growth in incoming edge weight, allowing S1 or S2 to be recognized before QDR collapses.

**XAI:** Return the influential events, neighbors, edges, and time intervals through temporal edge occlusion or a graph explainer. Box G then translates those entities back into Rule 1 or Rule 4 language.

**Risk:** Pseudo-events reconstructed from coarse poll snapshots are not equivalent to native controller events. If Phase 1 stores only aggregates, TGN loses much of its advantage.

### 3.2 EvolveGCN: graph snapshots with evolving parameters

**Role:** Box B plus Box D on regularly sampled graph snapshots. EvolveGCN uses an RNN to evolve GCN parameters rather than relying on persistent node embeddings [3_how_to_implement_each_of_the_ten_models[2]] [5].

**Input:** At each poll, construct a nine-node graph. Node matrix `X_t` contains the 10 per-switch features. The weighted adjacency contains W1-W12 and the separately typed logical edge. Feed a sequence of these snapshots into EvolveGCN-H and EvolveGCN-O variants as two internal configurations.

**Output:** Attach a node classifier to identify which switch is in S1-S3, a pooled graph classifier for global stage, and transition and dwell heads on the final hidden state.

**Concrete case:** EvolveGCN should detect the progression from distributed P1/P2 forwarding to a graph where incoming weight concentrates near the GBH path. It is also the correct stress-test candidate if later experiments move the rogue role among switches or change the node set.

**XAI:** Attribute the decision to node features and graph edges by removing or masking one edge or node history at a time. Report whether W values, QDR, entropy, or the logical edge caused the largest probability change.

**Risk:** Your production topology is fixed at nine switches. EvolveGCN's changing-node advantage may not justify its training complexity unless you randomize topology or rogue-node location during experimentation.

### 3.3 Graph WaveNet: the strongest fixed-topology snapshot candidate

**Role:** Box B and short-horizon Box D. Graph WaveNet learns an adaptive dependency matrix and combines it with stacked dilated temporal convolutions whose receptive field grows with depth [3_how_to_implement_each_of_the_ten_models[3]] [6].

**Input:** Use `[batch, channels, nodes, history]`. The native ten switch features become channels. Broadcast `path_ratio` as global context. To preserve W1-W12 without changing the model beyond recognition, test two declared variants: `GWNet-static`, using physical adjacency plus edge-derived node channels, and `GWNet-dynamic`, using the latest normalized W matrix as an additional graph support.

**Output:** Replace the original continuous traffic-forecasting output with current-stage logits and multi-horizon stage logits for `t+1...t+H`. Convert the first predicted transition horizon into estimated sojourn remaining.

**Concrete case:** During S2, the adaptive dependency matrix may learn that s5, s6, s7, and s8 interact more strongly than physical adjacency alone suggests. The dilated temporal stack can then distinguish a sustained concentration from a one-poll burst.

**XAI:** Return the learned adjacency, edge-support ablation, and temporal saliency. Do not present the adaptive adjacency as proof of a causal relationship; validate it by masking the reported edge and measuring the output change.

**Risk:** Vanilla Graph WaveNet was evaluated on traffic networks [3_how_to_implement_each_of_the_ten_models[3]] [6], not attack stages. Your dynamic-edge extension must be reported as an adaptation and ablated against the static version.

### 3.4 Temporal Fusion Transformer: interpretable multi-horizon progression

**Role:** Box B, Box D, and a major contributor to Box G. TFT combines multi-horizon forecasting with interpretable temporal information [3_how_to_implement_each_of_the_ten_models[4]] [4].

**Input:** Create a per-switch sequence. Static covariates may include switch degree, path membership, and topology role, but never `is_GBH` or a DPID encoding that makes s8 trivially identifiable. Observed time-varying inputs are the 10 features, W1-W12 summaries, and path ratio. Known future inputs should be limited to genuinely known values such as poll index or scheduled traffic regime.

**Output:** Use a classification head for current stage, horizon-specific softmax heads for future stages, and quantile heads for remaining sojourn. Variable-selection weights and temporal attention become part of the attribution payload. A maintained implementation is available through `pytorch-forecasting` [3_how_to_implement_each_of_the_ten_models[5]] [25].

**Concrete case:** TFT can show that a TG spike is important for immediate S1, while declining entropy and path-ratio collapse become more important for predicted S2 and QDR dominates later S3 horizons.

**XAI:** Combine variable-selection weights, attention, feature occlusion, and the rule table. Attention alone should not be the final explanation.

**Risk:** TFT can overfit a small synthetic dataset. Keep hidden size and attention depth modest, use early stopping, and compare its calibration and real-data gap against simpler baselines.

### 3.5 TimesNet: periodic and multi-scale stage signatures

**Role:** Direct Box B classification plus Box D forecasting. TimesNet transforms one-dimensional series into two-dimensional tensors based on multiple discovered periods [3_how_to_implement_each_of_the_ten_models[6]] [11] and supports classification, forecasting, and anomaly detection [3_how_to_implement_each_of_the_ten_models[6]] [11].

**Input:** Feed causal network or per-switch windows. Retain channels for route pressure, FII, TG, CC, CDA, QDR, entropy, skewness, kurtosis, divergence, edge weights, and path ratio. Tune the number of TimesBlocks and dominant periods only on validation runs.

**Output:** Use its classification pathway for current stage. Add next-stage and remaining-time heads to the final TimesBlock representation.

**Concrete case:** If probe bursts, route recalculation, and polling produce repeatable multi-scale patterns, TimesNet may separate an S1 burst from a persistent S2 concentration by modeling within-period and between-period variation.

**XAI:** Report the selected periods and use channel and time-block occlusion to test whether the reported periodic component actually changes the prediction.

**Risk:** The model assumes useful multi-periodicity. A deterministic synthetic generator can accidentally make every attack stage periodic, yielding excellent synthetic performance that disappears in Mininet. Randomize attack onset, stage duration, traffic rate, and poll jitter.

### 3.6 PatchTST: long-context self-supervised representation learning

**Role:** Box B and Box D when long histories are available. PatchTST splits time series into subseries patches and processes each channel independently with shared Transformer weights [3_how_to_implement_each_of_the_ten_models[7]] [15]. It supports forecasting and self-supervised representation learning [3_how_to_implement_each_of_the_ten_models[7]] [15].

**Input:** Form `[batch, channels, history]` tensors. Pretrain by masking patches from all available unlabeled normal and attack telemetry, then fine-tune on S0-S3 labels. Tune patch length and stride against the poll interval and observed stage durations.

**Output:** Pool the final patch embeddings for current-stage and transition heads. Use horizon-specific probabilities or a survival head for remaining time.

**Concrete case:** PatchTST can learn a long pre-attack baseline and identify that the current QDR or entropy patch deviates from the preceding routing regime, rather than responding to a single noisy poll.

**XAI:** Return patch-level integrated gradients or occlusion scores, then aggregate them into named feature and time ranges.

**Risk:** Native channel independence reduces direct interaction between QDR, CC, entropy, edge imbalance, and path ratio. Add a small cross-channel fusion layer after patch encoding, and ablate it. Otherwise the model may miss the cross-feature relationships central to blackhole behavior.

### 3.7 Neural CDE: irregular and missing controller telemetry

**Role:** Box B and continuous-time Box D. Neural CDE is specifically applicable to partially observed, irregularly sampled multivariate time series [3_how_to_implement_each_of_the_ten_models[8]] [3].

**Input:** For each session, provide timestamped observations, value channels, observation masks, and time-since-last-observation. Construct a continuous interpolation of the causal history and solve for latent state `z(t)`.

**Output:** Read `z(t)` at every poll for current stage, next transition, and hazard or sojourn parameters. A `torchcde` implementation provides GPU-capable differentiable solvers and supports adjoint backpropagation for improved memory efficiency [3_how_to_implement_each_of_the_ten_models[9]] [22].

**Concrete case:** If Ryu misses one port-stat response or polling intervals vary under controller load, Neural CDE can model the continuous trajectory without forward-filling every missing observation as if it were measured.

**XAI:** Use path-feature ablation, integrated gradients over the interpolated path, and counterfactual removal of observation channels.

**Risk:** With perfectly regular, complete polling, a CDE solver may add latency without improving detection. Record solver evaluations, p95 inference time, and numerical failures as operational metrics.

### 3.8 Deep nonlinear state-space model: latent attack dynamics

**Role:** Box B, Box D, anomaly scoring, and uncertainty estimation. Structured Inference Networks jointly learn a generative model and recurrent variational posterior for nonlinear state-space models [3_how_to_implement_each_of_the_ten_models[10]] [8]. The cited work reports higher held-out likelihood from the structured posterior [3_how_to_implement_each_of_the_ten_models[10]] [8].

**Input:** Use per-switch or network-wide sequences. Define latent state `z_t`, transition `p(z_t | z_t-1)`, emission `p(x_t | z_t)`, and inference model `q(z_t | x_1:t)`. Include the four-stage label only in supervised heads, not as an observed generative input at runtime.

**Output:** Current-stage logits come from `z_t`; next-stage and sojourn distributions come from the learned transition. Negative log likelihood or reconstruction error can supply an auxiliary anomaly signal.

**Concrete case:** The latent trajectory can represent a gradual move from normal routing to route pressure and concentration even when individual observed features are noisy or missing.

**XAI:** Decode latent changes back into feature-space contributions and report both posterior uncertainty and feature ablations.

**Risk:** Unsupervised latent states will not automatically align with S0-S3. Anchor them with supervised stage and transition losses, monitor posterior collapse, and compare the full model against a deterministic recurrent encoder.

### 3.9 Mamba: efficient long-session sequence modeling

**Role:** Box B and Box D for long histories. Mamba makes state-space parameters input-dependent so that the model can selectively propagate or forget information [3_how_to_implement_each_of_the_ten_models[11]] [12]. Its sequence complexity scales linearly, and the paper reports fast inference on the modalities it evaluated [3_how_to_implement_each_of_the_ten_models[11]] [12].

**Input:** Tokenize each causal network snapshot. The simplest token is the 103-value network vector; a better adaptation first encodes the nine nodes and 12 edges, then passes one pooled network token per poll to causal Mamba blocks.

**Output:** Apply current-stage, next-stage, and sojourn heads to the final state. Keep the model causal; bidirectional processing would leak future polls during online detection.

**Concrete case:** Mamba can retain the earlier S1 probe burst while processing a long S2 session, then combine that memory with later QDR decline to predict S3.

**XAI:** Use time-token and feature-group occlusion. Report which past polls are forgotten or retained only as an interpretation hypothesis, then verify it by perturbation.

**Risk:** Mamba has no native graph inductive bias. A flat 103-channel token may learn the fixed column order rather than routing structure. Compare flat Mamba with a graph-pre-encoded Mamba and test on changed traffic patterns or rogue-node placement.

### 3.10 DeepHit: next-stage and time-to-transition specialist

**Role:** Primarily Box D. DeepHit directly learns a survival-time distribution and allows risk relationships to change over time [3_how_to_implement_each_of_the_ten_models[12]] [2]. It supports multiple competing events [3_how_to_implement_each_of_the_ten_models[12]] [2].

**Input:** Generate one landmark record at selected polls. The covariate is the causal history embedding. The event type is the next transition; event time is the number of polls until it occurs. Runs ending before a transition are censored rather than mislabeled as long sojourns.

Use stage-conditional event sets:

- S0: transition to S1 versus censoring or explicit session end.
- S1: progression to S2 versus explicit deactivation or mitigation.
- S2: progression to S3 versus mitigation or rerouting.
- S3: recovery or termination, if these outcomes are modeled.

**Output:** Convert the event-time probability mass function into `next_stage_probabilities`, expected remaining polls, prediction intervals, and confidence. Add an auxiliary MLP current-stage head on the shared history encoder only for the unified comparison. The original DeepHit repository is available publicly [3_how_to_implement_each_of_the_ten_models[13]] [24].

**Concrete case:** At S1, DeepHit can estimate both the probability of reaching S2 and when it is likely to happen, which is more useful to Box E than an uncalibrated next-stage label alone.

**XAI:** Explain the cumulative incidence of each transition with feature occlusion or DeepSHAP, then render the strongest feature-time contributions through Box G.

**Risk:** DeepHit should not replace Box B by itself. Its value must be judged mainly on transition discrimination, timing error, calibration, and risk-alert lead time.

**Decision-ready insight:** TGN, EvolveGCN, and Graph WaveNet model topology directly; TFT, TimesNet, PatchTST, Neural CDE, and Mamba model telemetry histories; Deep SSM models latent dynamics; DeepHit models transition risk and timing. They should share outputs, not identical inputs.

## 4. Train All Ten With Comparable Targets and Losses

Use a shared multi-task objective wherever the backbone supports it:

```text
L_total =
    lambda_stage * CE_or_focal(current_stage)
  + lambda_next  * masked_CE(next_stage)
  + lambda_time  * sojourn_NLL_or_survival_loss
  + lambda_ord   * ordinal_distance_penalty
  + lambda_rule  * optional_rule_consistency
```

Tune the lambda values on validation runs. Do not hardcode the same weights for every model if one native objective requires a different magnitude. Report both the total loss and every component.

The current stage head predicts S0-S3. The transition head predicts only valid destinations under the current stage and explicit controller actions. The time head predicts either Weibull parameters, discrete hazards, quantiles, or DeepHit's probability mass function. Convert each native output into the common remaining-polls and interval representation before evaluation.

Train with causal windows only. If a window ends at poll `t`, its current-stage target is the state at `t`; next-stage and sojourn labels may be derived from later ground truth, but later feature values must never enter the input. Fit normalizers, feature selectors, synthetic augmentation, and calibration exclusively on training or validation data.

Class imbalance should be handled with class-weighted cross-entropy, focal loss, or balanced sampling at the run/session level. Do not randomly oversample heavily overlapping S1 windows, since almost identical copies can dominate training. F1 is the harmonic mean of precision and recall [4_train_all_ten_with_comparable_targets_and_losses[0]] [13], but per-class F1 and macro F1 are both needed because S0 may be much more common than S1-S3.

Use the same hyperparameter-search budget, number of random seeds, early-stopping policy, and maximum history information for all candidates. Parameter count need not be identical, but compare a small and medium configuration for each family. The baseline suite should contain:

1. Random Forest for current stage.
2. FT-Transformer for current stage.
3. The locked feature-weighted semi-Markov model for transition and sojourn.
4. A simple GRU or temporal MLP to show whether complexity actually helps.

**Decision-ready insight:** the semi-Markov model is not discarded. It becomes both the Box D baseline and, if useful, a calibration or smoothing layer over a complex backbone's transition probabilities.

## 5. Compare the Models and Select the Final Five

No actual final five can be named yet because no common benchmark results were supplied. The defensible process is a two-stage selection: eligibility gates first, weighted ranking second.

### Test protocol

1. Split complete runs into train, validation, synthetic test, and real Mininet/Ryu test.
2. Create windows only after splitting.
3. Use at least several random seeds and aggregate metrics by run, not by correlated row.
4. Tune on validation only.
5. Freeze the model and confidence calibration.
6. Evaluate once on synthetic holdout and untouched real runs.
7. Bootstrap confidence intervals by run.
8. Repeat a shifted test with unseen traffic rate, attack start time, stage duration, selective dropping, probe intensity, and normal congestion.

| Dimension | Required metrics | Why it matters |
|---|---|---|
| Current-stage detection | Macro F1, per-stage precision/recall/F1, balanced accuracy, MCC, confusion matrix | Prevents S0 prevalence from hiding weak S1 or S2 detection. |
| Ordered-stage correctness | Mean absolute stage error, quadratic weighted kappa, catastrophic jump rate | Predicting S3 for S0 is worse than predicting adjacent S1. |
| Transition prediction | Next-stage macro F1, S2/S3 AUPRC, transition NLL, multiclass Brier score | Measures both discrimination and probability quality. |
| Timing | MAE in polls/seconds, median absolute error, concordance, integrated Brier score, interval coverage | Tests the semi-Markov or survival claim, not just the next label. |
| Early warning | S1 lead time, S2-to-S3 lead time, detection delay, missed attacks | Rewards models that act before packet absorption. |
| Alert utility | Attack-event recall, false alerts per hour, duplicate alerts per session, time in HIGH/CRITICAL before S3 | Measures the full Boxes E-F outcome. |
| Robustness | Worst-scenario macro F1, real-data score, synthetic-to-real drop, missing-poll degradation | Exposes simulator-rule overfitting. |
| Operations | p50/p95 latency, events or windows per second, peak RAM/VRAM, parameter count | Ensures Ryu-side deployment is feasible. |
| Explanation | Fidelity under top-feature removal, stability across nearby polls, rule agreement, analyst readability | Prevents persuasive but unfaithful narratives. |

Probability calibration must not be inferred from accuracy. Proper scoring rules such as Brier score and log loss evaluate probability quality together with discrimination and uncertainty [5_compare_the_models_and_select_the_final_five[0]] [20]. Calibrate each candidate on validation data, then freeze the calibrator before the test run.

### Recommended 100-point ranking

| Component | Weight |
|---|---:|
| Current-stage macro F1 | 20 |
| S1 recall and S2 recall | 10 |
| Ordered-stage score | 5 |
| Next-stage macro F1 and S2/S3 AUPRC | 20 |
| Sojourn and event-time performance | 10 |
| Probability calibration | 10 |
| End-to-end alert utility | 10 |
| Real-data robustness and shift resistance | 10 |
| Latency and resource efficiency | 5 |
| **Total** | **100** |

Normalize every component using a predeclared rule, average across seeds, and rank by the lower confidence bound rather than the most favorable single run. Explanation fidelity should be an eligibility gate rather than a small score that excellent accuracy can overwhelm.

### Hard eligibility gates

A candidate should be ineligible if any of these applies:

- It was tested only on synthetic data.
- It does not beat or match the relevant simple baseline within uncertainty.
- It has unacceptable S1 or S2 recall even if overall accuracy is high.
- Its p95 inference latency exceeds the poll interval or controller budget.
- It produces poorly calibrated confidence that destabilizes Boxes E-F.
- It relies on DPID, fixed attack timing, or future information.
- Its explanation fails a perturbation-fidelity test.

After these gates, take the top five weighted scores. If two models are statistically tied, prefer the one with lower latency, smaller synthetic-to-real drop, and simpler maintenance.

### Provisional implementation order, not final ranking

| Priority | Model | Reason for early implementation |
|---:|---|---|
| 1 | TGN | Best event-level match to route attraction and changing edge traffic. |
| 2 | Graph WaveNet | Strong fixed-topology spatial-temporal candidate. |
| 3 | TFT | Multi-horizon prediction plus useful feature-selection outputs. |
| 4 | Mamba | Efficient long-history sequence candidate. |
| 5 | DeepHit | Native next-transition and time-to-transition specialist. |
| 6 | EvolveGCN | Important dynamic-graph comparator. |
| 7 | Neural CDE | Essential if real polling is irregular or incomplete. |
| 8 | PatchTST | Strong self-supervised long-context candidate. |
| 9 | TimesNet | Valuable if genuine multi-periodicity exists. |
| 10 | Deep SSM | High research value but highest training and interpretation risk. |

This table is a build-priority judgment. It is not evidence that the first five will achieve the best metrics.

**Decision-ready insight:** the final five must be selected from real, grouped, shifted tests. Until then, call them candidates or an implementation shortlist, not winners.

## 6. Revised Build Plan and Repository Structure

Implement shared infrastructure before model code. Otherwise each student or developer will create incompatible windows, labels, and metrics.

| Step | Deliverable | Key acceptance test |
|---:|---|---|
| 1 | Fix Phase 1 measurement bugs | Entropy, STD, RREQ/RREP, W1-W12, and path ratio change under controlled attacks for physically explainable reasons. |
| 2 | Add IDs and event labels | Every row can be traced to run, poll, switch, attack activation, and controller ground truth. |
| 3 | Build synthetic generator v2 | Randomizes traffic, timing, duration, probe intensity, drop policy, and benign congestion without deterministic label leakage. |
| 4 | Build split manifest | No run, activation, seed, or overlapping window crosses partitions. |
| 5 | Build canonical adapters | Event graph, graph snapshot, regular sequence, irregular sequence, and survival records produce tested shapes. |
| 6 | Freeze baselines | RF/FT-Transformer and semi-Markov scores are reproducible. |
| 7 | Implement graph family | TGN, EvolveGCN, Graph WaveNet. |
| 8 | Implement sequence family | TFT, TimesNet, PatchTST, Neural CDE, Mamba. |
| 9 | Implement probabilistic family | Deep SSM and DeepHit. |
| 10 | Add common calibration and outputs | All models satisfy `ModelOutput`. |
| 11 | Add Box G attribution providers | Attributions map to named node, edge, feature, and time evidence. |
| 12 | Run synthetic then real benchmark | Produce one locked leaderboard with confidence intervals and eligibility status. |

A practical repository layout is:

```text
intrusex_bh/
  data/
    schema.py
    split_manifest.py
    synthetic_generator.py
    adapters/
      event_graph.py
      graph_snapshot.py
      regular_sequence.py
      irregular_sequence.py
      survival_landmarks.py
  models/
    base.py
    tgn_adapter.py
    evolvegcn_adapter.py
    graph_wavenet_adapter.py
    tft_adapter.py
    timesnet_adapter.py
    patchtst_adapter.py
    neural_cde_adapter.py
    deep_ssm_adapter.py
    mamba_adapter.py
    deephit_adapter.py
  heads/
    stage.py
    transition.py
    sojourn.py
  intelligence/
    sessions.py
    risk.py
    thresholds.py
    attribution.py
    narratives.py
  evaluation/
    metrics.py
    calibration.py
    robustness.py
    leaderboard.py
  configs/
  tests/
```

Store configuration, code commit, split manifest, random seed, fitted scaler, model checkpoint, calibration object, and metric output for every run. The model repository alone is not enough for reproducibility.

**Decision-ready insight:** make data splits and adapters shared platform components. Model development can then proceed in parallel without producing ten incomparable experiments.

## 7. Failure Cases the Benchmark Must Deliberately Include

| Failure case | What a weak model learns | Required mitigation |
|---|---|---|
| s8 is always malicious | Memorizes DPID or position | Exclude raw identity from learned inputs and rotate rogue placement in synthetic robustness tests where feasible. |
| Attack always starts at the same poll | Learns elapsed time | Randomize start, duration, and deactivation. |
| S1 always has the same TG spike | Learns generator threshold | Vary probe patterns and add benign route-recalculation spikes. |
| S2 always lowers entropy monotonically | Confuses congestion with dominance | Add flash crowds, link failures, and legitimate path imbalance as hard negatives. |
| S3 always drops every packet | Misses selective blackholes | Include probabilistic, burst, and destination-selective dropping. |
| Global edge values are repeated in every switch row | Overweights duplicated context | Keep a single global tensor or use a declared fusion layer. |
| Random row splitting | Tests near-duplicates of training windows | Group by complete run and build windows after splitting. |
| Synthetic scaler applied to real data unchanged | Hides distribution shift | Fit on training only and report real out-of-range rates; recalibration must use a separate adaptation set. |
| SHAP used identically for every backbone | Produces slow or inappropriate explanations | Use model-specific attribution behind one common Box G interface. |
| Risk score evaluated separately from detection | Hides threshold instability | Evaluate alerts after Boxes E-F, including false alerts per hour and lead time. |

The greatest scientific risk is not choosing the wrong deep architecture. It is creating a synthetic generator whose labels, stage durations, edge dominance, and feature trajectories are so deterministic that every high-capacity model obtains near-perfect results. The train-synthetic/test-real procedure is therefore central, not optional [7_failure_cases_the_benchmark_must_deliberately_include[0]] [16].

A second tension concerns explainability. RF can use exact tree-specific attribution methods, while graph and deep sequence models require edge, time, patch, or latent-state explanations. Box G should be renamed internally from a `SHAP wrapper` to an `Attribution Provider`; SHAP may remain one implementation, but every provider must output a common evidence schema and pass perturbation testing.

**Decision-ready insight:** design the negative and shifted scenarios before training. Otherwise the leaderboard will rank simulator memorization rather than blackhole intelligence.

## Synthesis

The ten candidates differ along four decisive dimensions. First, their **mechanism** differs: TGN operates on timed events, EvolveGCN and Graph WaveNet on graph snapshots, TFT/TimesNet/PatchTST/Neural CDE/Mamba on temporal sequences, Deep SSM on latent dynamics, and DeepHit on event-time risk. Second, their **scope** differs: graph models can localize a suspicious switch or edge, while flattened sequence models more naturally describe network-level progression. Third, their **time horizon** differs: TGN and Graph WaveNet are strong for immediate structural change, TFT and Mamba for multi-poll evolution, and DeepHit for calibrated transition timing. Fourth, their **trade-offs** differ: Neural CDE handles irregular sampling but adds solver cost; PatchTST supports self-supervised pretraining but weakens native cross-channel interaction; Deep SSM supplies uncertainty but risks latent-state misalignment.

The non-obvious conclusion is that the best deployment may not be one universal winner. A graph model may be best for Box B localization, while DeepHit or TFT may be best for Box D timing. Nevertheless, your request for a final five can be handled fairly by giving every model the common multi-task interface and ranking the complete pipeline. The final five should be the five eligible models with the strongest real-data lower-confidence-bound scores, not necessarily one model from each family.

The current architecture-fit hypothesis is:

1. **TGN** should excel if true controller events are retained.
2. **Graph WaveNet** should excel on fixed-interval nine-node snapshots.
3. **TFT** should offer the most accessible multi-horizon explanation package.
4. **Mamba** should be competitive when sessions become long enough for linear sequence scaling to matter.
5. **DeepHit** should add the clearest value for transition probability and remaining time.

EvolveGCN could displace one of these if topology or rogue-node location varies. Neural CDE could displace Mamba if missing and irregular polling dominate. PatchTST could displace TFT after self-supervised pretraining on substantial unlabeled real telemetry. TimesNet could rise if real periodic behavior exists. Deep SSM could become valuable if uncertainty and missing-data likelihood are more important than operational simplicity.

Therefore, implement all ten behind the shared adapter, but make no final-five claim from synthetic metrics alone. The locked IntruSex-BH architecture remains valid; it needs a causal history adapter, common multi-task heads, model-specific attribution providers, and a stricter real-data selection protocol.

## References

1. *GitHub - PatchTST/PatchTST: An offical implementation of ...*. https://github.com/PatchTST/PatchTST
2. *A Deep Learning Approach to Survival Analysis With Competing Risks*. http://ojs.aaai.org/index.php/AAAI/article/view/11842
3. *Neural Controlled Differential Equations for Irregular Time Series*. https://arxiv.org/abs/2005.08926
4. *Temporal Fusion Transformers for Interpretable Multi-horizon ...*. https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting
5. *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic ...*. https://ojs.aaai.org/index.php/AAAI/article/view/5984
6. *Graph WaveNet for Deep Spatial-Temporal Graph Modeling IJCAI https://www.ijcai.org › proceedings › 2019*. http://ijcai.org/proceedings/2019/264
7. *Towards bridging the synthetic-to-real gap in quantitative photoacoustic tomography via unsupervised domain adaptation - ScienceDirect*. http://sciencedirect.com/science/article/pii/S221359792500059X
8. *Structured Inference Networks for Nonlinear State Space Models*. https://arxiv.org/abs/1609.09869
9. *Temporal Graph Networks for Deep Learning on Dynamic Graphs*. https://arxiv.org/abs/2006.10637
10. *sksurv.metrics.integrated_brier_score#*. https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.integrated_brier_score.html
11. *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*. https://arxiv.org/abs/2210.02186
12. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. https://arxiv.org/abs/2312.00752
13. *f1_score — scikit-learn 1.9.0 documentation*. http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
14. *Synthetic-to-Real Object Detection using YOLOv11 and Domain Randomization Strategies*. https://arxiv.org/html/2509.15045v1
15. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. https://arxiv.org/abs/2211.14730
16. *Synthetic Data: Revisiting the Privacy-Utility Trade-off*. https://arxiv.org/html/2407.07926v2
17. *Weighted kappa measures for ordinal multi-class ...*. https://www.sciencedirect.com/science/article/pii/S1568494623000388
18. *Finite-Horizon Quickest Change Detection Balancing Latency ...*. http://arxiv.org/pdf/2511.12803
19. *GroupKFold — scikit-learn 1.9.0 documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
20. *1.16. Probability calibration — scikit-learn 1.9.0 documentation*. https://scikit-learn.org/stable/modules/calibration.html
21. *shap.TreeExplainer — SHAP latest documentation*. https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
22. *GitHub - patrick-kidger/torchcde: Differentiable controlled ...*. https://github.com/patrick-kidger/torchcde
23. *GitHub - twitter-research/tgn: TGN: Temporal Graph Networks · GitHub*. http://github.com/twitter-research/tgn
24. *GitHub - chl8856/DeepHit: DeepHit: A Deep Learning Approach ...*. https://github.com/chl8856/DeepHit
25. *TemporalFusionTransformer — pytorch-forecasting documentation*. https://pytorch-forecasting.readthedocs.io/en/v1.2.0/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html
