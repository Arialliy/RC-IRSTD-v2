# RC-IRSTD-v2: Risk-Controlled Infrared Small Target Detection

RC-IRSTD-v2 is a research implementation of a two-stage system for
cross-domain infrared small-target detection. The proposed system combines the
baseline-preserving **RC-MSHNet** detector with a dual-monotone **RiskCurve**
operating-point selector. The detector backbone and Scale/Location Sensitive
lineage originate from the upstream MSHNet project; the upstream notice and
reported reference result are retained below with explicit attribution.

## Code-only GitHub release

This GitHub release contains the implementation, configuration contracts,
launch/evaluation utilities, and tests needed to inspect and extend the
RC-IRSTD-v2 research code. It intentionally excludes datasets, checkpoints,
score maps, experiment outputs, audit archives, logs, caches, and other
machine-generated artifacts.

Included source areas are `rc_irstd/`, `risk_curve/`, `evaluation/`,
`data_ext/`, `losses/`, `certification/`, `model/`, `scripts/`, `tests/`,
`configs/`, `tools/`, `utils/`, and the root Python entry points. To run an
experiment, provide the datasets and any required frozen evidence/checkpoints
locally; do not commit them to the code branch.

Before packaging this release, the full development tree passed `1105` tests
with `27` skips. Tests that verify frozen run artifacts require the separate
internal experiment archive and may not be runnable from the code-only
checkout by itself.

## Current AAAI-27 execution status

The latest frozen audit state is:

| Stage | Status | Decision | Label/data boundary |
|---|---|---|---|
| Phase 2 detector gate | `completed` | `HOLD` | Six matched detector/control runs are frozen; RiskCurve has not started. |
| Phase 3 source-only inner LODO Tier 1 | `tier1_completed` | `HOLD` | Source official-train labels were used for the pseudo-target audit; outer-target images and labels were not used. |
| Phase 3 raw-logit rescue | `completed` | `RESCUE_GO_TIER2` | Exact source-only raw-logit diagnostics authorized Tier 2; outer-target access remained closed. |
| Phase 3 Tier 2 raw-logit gate | `completed` | `TIER2_HOLD` | The frozen Tier 2 comparison did not authorize Tier 3 or outer-target access. |
| Phase 3 Tier2R component rescue | `tier2r_exact_gate_completed` | `TIER2R_HOLD` | No candidate was selected, the component claim was dropped, and source Tier 3 plus outer-target access remain unauthorized. |
| Tier2S factorized causal audit | `tooling_only` | `NOT_RUN` | Source-only diagnostic tooling is included. The GPU0/1 relocation path is an unregistered candidate and is not evidence of execution or authorization. |

Tier 1 entered `scientific_hold_or_rescue`; the frozen raw-logit rescue then
authorized a source-only Tier 2 evaluation. Tier 2 and the preregistered Tier2R
component rescue both completed, but the final decision remains
`TIER2R_HOLD`: no candidate, source Tier 3 design, RiskCurve stage, outer-target
image access, or outer-target label access is authorized. The `HOLD_RC_MSHNET_GATE`,
`PHASE3_SOURCE_TIER1_HOLD`, and `HOLD_PHASE3_TARGET_LABEL_ACCESS` sentinels are
fail-closed controls and must not be removed merely to advance the pipeline.

The machine-verifiable run state lives in a separate internal archive and is
not bundled with this code-only release. The public checkout retains the
executable method and policy contracts under `configs/` together with their
corresponding validation tests. Supplying an audit archive later must not
change or relabel the frozen `TIER2R_HOLD` decision.

The current frozen method identities are defined by
[`configs/rc_mshnet_aaai27_method_contract.yaml`](configs/rc_mshnet_aaai27_method_contract.yaml)
and
[`configs/rc_v2_aaai27_main.yaml`](configs/rc_v2_aaai27_main.yaml).

## Upstream MSHNet notice 📰

This repository is an RC-IRSTD-v2 research fork based on the public MSHNet
implementation. The numbers and weights in this introductory table are reported
by the upstream MSHNet authors; they have not been freshly reproduced by the
RC-IRSTD-v2 pipeline.

| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 67.87 | 92.86 | 8.88 | [new_weights](https://drive.google.com/file/d/1CSDwQG8xg7hv0_oGKa4NCEWUiMRU7eIs/view?usp=sharing) |

## Overview

The rendered overview asset from the full experiment package is intentionally
omitted from the code-only branch. The executable architecture is defined by
the modules and configuration contracts listed above.

## Upstream paper summary

The detector backbone is based on the official implementation of the CVPR 2024
paper [Infrared Small Target Detection with Scale and Location Sensitivity](https://arxiv.org/abs/2403.19366).

The upstream authors introduce a Scale and Location Sensitive (SLS) loss and a
multi-scale head for a plain U-Net. Their paper reports the following
contributions; these statements describe the upstream work, not new RC-IRSTD-v2
results:

1. An SLS loss intended to improve scale and location sensitivity.

2. A multi-scale detection head built on a U-Net backbone.

3. Upstream experiments reporting gains from applying SLS to other detectors.

## Environment

Install a PyTorch and torchvision build that matches the host CPU/CUDA runtime
from the [official PyTorch selector](https://pytorch.org/get-started/locally/),
then install the remaining dependencies:

```bash
"${PYTHON_BIN:-python3}" -m pip install -r requirements.txt
```

`requirements.txt` deliberately does not pin a CUDA build. Set
`PYTHON_BIN` to the interpreter for your environment when it is not `python3`.

MSHNet pools four times, so `--base-size` and `--crop-size` must be positive
multiples of 16. The legacy loader accepts the original split files and
`img_idx/train_*.txt` / `img_idx/test_*.txt`. NUAA-SIRST
`*_pixels0.png` masks are resolved explicitly; XML annotations are never
treated as images.

### Frozen splits and the NUAA `Misc_111` resolution quirk

The manifests under each local dataset's `img_idx/` directory are the split
authority. The repository does not create a validation split or repartition
the data; the legacy `val` loader name is only an alias for the frozen test
manifest and must not be used for paper-model selection.

The distributed NUAA-SIRST `Misc_111` image is `325 x 220`, while its mask is
`592 x 400`. Loaders keep the image coordinate system and apply a corrected,
guarded compatibility rule: when the relative width/height-ratio error is at
most 1%, resize the mask to the image with nearest-neighbour interpolation;
otherwise fail. This fixes the coordinate misalignment caused by BasicIRSTD's
evaluation-time top-left crop. The `Misc_111` error is about 0.185%. Raw dataset
files are never rewritten. Native score exports record the original mask size,
relative error, policy, and aligned sample IDs in the version-3 integrity
manifest.

Audit all frozen manifests and raster pairs before a paper run:

```bash
"${PYTHON_BIN:-python3}" scripts/audit_dataset_splits.py \
  --dataset-dir datasets/NUAA-SIRST \
  --dataset-dir datasets/NUDT-SIRST \
  --dataset-dir datasets/IRSTD-1K \
  --output artifacts/dataset_split_audit.json
```

An eligible resolution alignment is retained as a warning in the report;
missing/ambiguous rasters, incompatible aspect ratios, split-ID overlap, and
exact cross-split image or image-mask-pair duplicates remain hard failures.

## AAAI-27 method identity and evaluation protocol

The proposed method is a composition, not a renamed baseline. **MSHNet +
StableSLS-v1** is the canonical detector baseline, **MSHNet-FT + StableSLS-v1**
is the matched fine-tuning control, and **RC-MSHNet + StableSLS-v1** is the
proposed detector. RC-MSHNet inherits canonical MSHNet and uses zero-residual
initialization, then adds SN-LCP, CS-CCF, and RP-RGF as independently ablatable
extensions.

The complete proposed system is **RC-IRSTD = RC-MSHNet + dual-monotone
RiskCurve**. Frozen FP32 raw logits from RC-MSHNet are consumed by
`risk_curve.monotone_curve_predictor.RiskCurvePredictor`, which predicts the
full pixel-risk curve and a conservative monotone component-risk curve over a
shared finite threshold grid. A zero-label selector converts those curves into
a threshold index or an explicit reject action. An optional few-shot CRC stage
may then learn one shared rank offset from a separate labelled calibration
set. Under the current frozen `TIER2R_HOLD`, RiskCurve, source Tier 3, and
outer-target evaluation have not been authorized or run.

`rc_irstd.models.calibrator.MonotoneBudgetCalibrator` is **RC-Direct**, a strong
direct-threshold baseline. It is not the proposed full-curve model and its
checkpoint must not be reported as the main RC-IRSTD-v2 result. The executable
method contract is recorded in
[`configs/rc_v2_aaai27_main.yaml`](configs/rc_v2_aaai27_main.yaml); the three
outer configurations select `method: risk_curve` for the main experiments.

The datasets expose frozen official `train` and `test` manifests only. In each
outer fold, pseudo/meta supervision is built exclusively from the two source
domains' official **train** images. The held-out target's official **test**
images and masks are not used to train the detector or risk-curve predictor.
Target masks are not consumed by deployment-statistics construction or the
zero-label selector; they are consumed only by the final metric/audit stage.
The legacy loader spelling `val` still maps to the official test manifest, but
it must never be interpreted as a newly created validation split or used for
model selection. Detector checkpoints are chosen by the pre-specified
fixed-last policy (`last.pt`), never by target-test labels.

With three domains, the intended full-system evaluation consists of three
outer leave-one-domain-out (LODO) folds:

| Outer target | Meta-sources | Proposed-system configuration |
|---|---|---|
| NUAA-SIRST | NUDT-SIRST, IRSTD-1K | `configs/pipeline_outer_nuaa_rc_mshnet_v4.yaml` |
| NUDT-SIRST | NUAA-SIRST, IRSTD-1K | `configs/pipeline_outer_nudt_rc_mshnet_v4.yaml` |
| IRSTD-1K | NUAA-SIRST, NUDT-SIRST | `configs/pipeline_outer_irstd_rc_mshnet_v4.yaml` |

These outer configurations are protocol contracts, not authorization to run.
While the latest Phase 3 decision is `TIER2R_HOLD`, do not launch the outer
folds or open outer-target labels. Inspect the frozen source-only gates first:

```bash
cat artifacts/aaai27/audit/phase3_source_lodo_gate/phase3_status.json
cat artifacts/aaai27/audit/phase3_source_lodo_gate/tier1_decision.json
```

The legacy `pipeline_outer_{nuaa,nudt,irstd}.yaml` files retain the earlier
canonical-MSHNet route for baseline and compatibility work; they are not the
RC-MSHNet full-system configurations.

Static image collections have no defensible frame order. Their main zero-label
evaluation is therefore deterministic, full-coverage 5-fold cross-fit on the
official target test set. The seeded permutation fixes the five folds. For
each held-out fold, exactly `A=32` images are then sampled deterministically
and **without replacement** from the union of the other four folds; only those
32 score/gray pairs form the adaptation statistic. The held-out fold in its
entirety is the query set, and the frozen fold action is mapped to every one of
its IDs. Rotating the held-out fold therefore evaluates every test image
exactly once. A run fails instead of silently shrinking `A` if a four-fold
complement contains fewer than 32 images. No mask is read during adaptation or
selection. This protocol is explicitly **transductive**; it supports neither a
causal claim nor a formal CRC certificate. Standard detector comparison is
reported separately on every official test image.

Only a dataset with trustworthy `sequence_id` and `frame_index` metadata may
use the separate temporal sensitivity protocol. That protocol uses
`A=32`, `E=1`, and `stride=33`, so adaptation and evaluation windows are
non-overlapping. The old `support=32`, `query=64`, `stride=96` direct-calibrator
pipeline is not the AAAI-27 main method. Optional few-shot CRC is run only when
labelled calibration and final-test IDs are explicitly disjoint and the
finite-sample feasibility check passes; otherwise its status must remain
`not_run` rather than being inferred from the zero-label experiment.

Each proposed-system fold uses the `output_dir` frozen in its v4 YAML. The
three roots are `outputs/aaai27/risk_curve/outer_{nuaa,nudt,irstd}_seed42/`,
with the following per-fold structure:

```text
outputs/aaai27/risk_curve/outer_<target>_seed42/
├── lodo/<PSEUDO_TARGET>/detector/last.pt
├── risk_curve_main/
│   ├── threshold_grid.npy
│   ├── curve_episodes/{train.npz,val.npz,manifest.json}
│   └── best.pt
├── direct_threshold_baseline/
│   ├── episodes/
│   └── checkpoints/last.pt          # isolated RC-Direct baseline
├── final_detector/last.pt
├── targets/<TARGET>/
│   ├── scores/
│   ├── standard_detection/threshold_sweep.csv
│   ├── risk_curve_main/
│   │   ├── deployment_statistics.npz
│   │   └── budgets/
│   │       └── <BUDGET>/{zero_selection.json,calibration_losses.npz,
│   │                    zero_label_evaluation.json}
│   └── direct_threshold_baseline/static_cross_fit.json
└── pipeline_summary.json
```

The exact artifact names written by a run are listed in its
`pipeline_summary.json`; do not substitute an RC-Direct checkpoint for the
risk-curve checkpoint when assembling a paper table.

When enabled (the formal outer configs enable it), RC-Direct is trained from
the same source-train pseudo domains and evaluated with the same seeded,
fixed-`A=32`, full-coverage static folds. Its threshold request uses the pixel
budget; the component budget is an independent realised-risk audit and is
reported explicitly as raw connected-component risk, not as the proposed
model's learned conservative curve.

Before a full run, execute the test suite. After source-train curve episodes
exist, run the two-epoch main-method trainer smoke explicitly:

```bash
PY="${PYTHON_BIN:-python3}"
make PYTHON="$PY" test
TRAIN_EPISODES=outputs/curve_episodes/train.npz \
VAL_EPISODES=outputs/curve_episodes/val.npz \
DEVICE=cuda:0 PYTHON_BIN="$PY" \
  bash scripts/launch_risk_curve_smoke.sh
```

`PYTHON_BIN="$PYTHON_BIN" make smoke` is retained as a legacy RC-Direct wiring
regression. Passing it does not validate the proposed risk-curve route.

`configs/pipeline.yaml` has three meta-sources but no unseen fourth target. Its
empty `final_targets` output is a meta-training artifact, not a final paper
result. Do not reuse a meta-source as a target to fill that gap.

### Documentation authority

The current YAML files, both frozen method contracts, the Phase 2/3 status and
decision JSON files, and each CLI's `--help` output are the execution authority.
README prose never overrides a `HOLD` sentinel or a machine-verifiable gate
decision. The following long documents are retained as historical design
provenance only and are **not executable runbooks**:

- `docs/complete_solution/RC-IRSTD_完整方案_模型_代码_训练启动.md`
- `docs/complete_solution/完整方案设计.md`

They may contain obsolete commands such as `train.py --config`, detector
`best.pt`, overlapping `stride=8`, or a direct threshold calibrator presented as
the proposed model. Do not copy those commands into an AAAI-27 main run. The
manual commands below expose individual stages for auditing and ablations; the
three outer YAMLs remain the complete main-experiment entry points.

## RC-IRSTD-v2 quickstart

Set the prepared interpreter once and materialize the shared threshold grid:

```bash
PY="${PYTHON_BIN:-python3}"
$PY -m risk_curve.threshold_grid --output artifacts/threshold_grid.npy
```

### 1. Shared export and threshold diagnostics

Use fixed `resize` inputs only for quick diagnosis and regression comparison:

```bash
$PY -m evaluation.export_score_maps \
  --dataset-dir datasets/NUAA-SIRST \
  --split test \
  --weight-path <DETECTOR_WEIGHT.pkl> \
  --output-dir outputs/score_maps/nuaa_resize \
  --spatial-mode resize --base-size 256 \
  --device cuda

$PY -m evaluation.threshold_sweep \
  --score-dir outputs/score_maps/nuaa_resize \
  --threshold-grid artifacts/threshold_grid.npy \
  --output outputs/curves/nuaa_resize.csv
```

Final paper and deployment metrics must be regenerated with
`--spatial-mode native --pad-multiple 16 --batch-size 1`. Native export crops
away padding before risk denominators are computed; resized pixels do not
preserve the physical meaning of false alarms per original pixel or megapixel.

All raw-mask consumers share `data_ext.mask_alignment.align_mask_to_image`.
For NUAA-SIRST `Misc_111`, this corrected compatibility rule computes the
relative width/height-ratio error first, rejects values above 1%, and only then
resizes the mask to the original PIL image size with
`PIL.Image.Resampling.NEAREST`. It replaces BasicIRSTD's evaluation-time
top-left crop, which leaves the mask in the wrong coordinate system. The call
happens before training augmentation, evaluation resize/padding, and score
export; formal manifests retain the original mask dimensions and alignment
evidence.

For a frozen checkpoint's formal native test evaluation, keep label-free and
labeled exports in different directories. Verify the label-free export first;
then generate the labeled export and run the strict evaluators:

```bash
$PY -m evaluation.standard_metrics \
  --score-dir outputs/score_maps/<detector>/scores_labeled \
  --threshold 0.5 \
  --output outputs/score_maps/<detector>/standard_metrics_0_5.json

$PY -m evaluation.threshold_sweep \
  --score-dir outputs/score_maps/<detector>/scores_labeled \
  --formal --expected-split-role test \
  --output outputs/curves/<detector>_test.csv
```

The formal sweep writes the bound provenance sidecar
`<detector>_test.csv.metadata.json`. Source pooled/worst operating points must
be selected from a complete official-train pseudo-target LODO collection with
`evaluation.source_operating_point`; an optional target curve is loaded only
after both source thresholds are frozen.

### 2. Optional detector-training ablation: balanced Tail-CVaR

The AAAI-27 method comparison keeps the detector identity explicit. The
following balanced Tail-CVaR trainer is a detector-training ablation; it does
not replace `RiskCurvePredictor` and its result must be reported separately
from the canonical MSHNet-detector route.

```bash
$PY -m scripts.train_multisource_tail \
  --source-dirs datasets/NUAA-SIRST datasets/NUDT-SIRST datasets/IRSTD-1K \
  --batch-per-domain 2 --epochs 400 \
  --lambda-tail 0.1 --lambda-miss 0.1 \
  --tail-q 0.01 --miss-q 0.2 --tail-gamma 10 \
  --save-dir repro_runs/rc_tail
```

Resume refuses changes to source order, spatial sizes, balanced batch,
optimization/loss hyperparameters, or the warm-stage contract. Start a new run
instead of silently mixing configurations.

The multi-source trainer defaults to `--checkpoint-selection fixed_last` for
paper runs. It writes the explicit inference artifact `last.pkl`, mirrors the
selected artifact to `weight.pkl`, retains `best_train_loss.pkl` for diagnostics
only, and uses `checkpoint.pkl` solely for complete-state resume. Source split
paths, hashes, and train/test disjointness are validated before training.

### 3. Build source-train pseudo-target episodes and train the risk curve

Export each inner-loop pseudo-target from its official `train` manifest, using
the matching spatial protocol and a detector trained without that
pseudo-target. Repeat `--source-dataset` during export for every detector
training domain; the episode builder rejects fold leakage or missing
provenance. In an outer fold, one source pseudo-domain supplies training
episodes and the other is the pre-specified held-out pseudo-domain for
validation. The outer target's official `test` split is absent from both
archives. A stage-level example is:

```bash
$PY -m risk_curve.build_curve_episodes \
  --score-map-dir <PSEUDO_TRAIN_SCORE_DIR> --pseudo-target pseudo_train \
  --score-map-dir <PSEUDO_VAL_SCORE_DIR> --pseudo-target pseudo_val \
  --threshold-grid artifacts/threshold_grid.npy \
  --adaptation-window 32 --evaluation-window 1 --stride 33 \
  --validation-domain pseudo_val \
  --output-dir outputs/curve_episodes

$PY -m risk_curve.train_curve_predictor \
  --train-file outputs/curve_episodes/train.npz \
  --val-file outputs/curve_episodes/val.npz \
  --output outputs/models/risk_curve_q90.pt \
  --quantile 0.90 --device cuda
```

The predictor learns both pixel log-risk and component log-risk across the
whole threshold grid. Component supervision uses the conservative monotone
upper envelope; the raw component curve remains in the archive for auditing.
The episode geometry is `A=32`, `E=1`, `stride=33`; `E>1` predicts aggregate
future risk and is diagnostic only. For static source collections this is a
pre-specified meta-training construction, not evidence of temporal causality.
Only genuinely ordered sequence data may attach a causal interpretation to the
same A/E contract. Selection and calibration reload the checkpoint contract
and reject grid or provenance mismatches.

### 4. Zero-label operating point

For the main static-image protocol, build mask-free complement-fold statistics
and predict one action for every held-out test image:

```bash
$PY -m risk_curve.build_deployment_statistics \
  --score-map-dir outputs/score_maps/target_test_native \
  --threshold-grid artifacts/threshold_grid.npy \
  --mode static-cross-fit --folds 5 --seed 42 --adaptation-window 32 \
  --output outputs/zero_label/static_crossfit_statistics.npz

$PY -m risk_curve.select_zero_label_threshold \
  --statistics-file outputs/zero_label/static_crossfit_statistics.npz \
  --curve-checkpoint outputs/models/risk_curve_q90.pt \
  --pixel-budget 1e-6 --component-budget 1.0 \
  --output outputs/zero_label/static_crossfit_selection.json \
  --device cuda
```

In each fold, the seed and fold index deterministically select exactly 32
adaptation IDs without replacement from the other four folds. The complete
held-out fold is the query set; no held-out ID contributes to that fold's
adaptation statistic, and every target-test image is queried exactly once.
Complement-fold statistics do not read masks, but they do use the unlabeled
target test collection, so the result must be labelled **static 5-fold
transductive cross-fit**. Zero-label selection is empirical and carries no
unconditional finite-sample or formal CRC guarantee. Build count curves only
after actions are frozen, then run the independent ID-bound audit:

```bash
$PY -m certification.build_calibration_losses \
  --score-dir outputs/score_maps/target_test_native \
  --threshold-grid artifacts/threshold_grid.npy \
  --pixel-budget 1e-6 --component-budget 1.0 \
  --loss-mode budget_violation \
  --output outputs/zero_label/target_test_count_curves.npz

$PY -m risk_curve.evaluate_zero_label \
  --zero-result outputs/zero_label/static_crossfit_selection.json \
  --count-curves outputs/zero_label/target_test_count_curves.npz \
  --output outputs/zero_label/static_crossfit_audit.json
```

For the separate temporal sensitivity experiment, first verify real sequence
metadata. Then build disjoint causal deployment statistics without reading
masks and predict one base index per future image:

```bash
$PY -m risk_curve.build_deployment_statistics \
  --score-map-dir outputs/score_maps/target_causal_native \
  --threshold-grid artifacts/threshold_grid.npy \
  --mode causal \
  --adaptation-window 32 --evaluation-window 1 --stride 33 \
  --output outputs/zero_label/deployment_statistics.npz

$PY -m risk_curve.select_zero_label_threshold \
  --statistics-file outputs/zero_label/deployment_statistics.npz \
  --curve-checkpoint outputs/models/risk_curve_q90.pt \
  --pixel-budget 1e-6 --component-budget 1.0 \
  --output outputs/zero_label/selection_adaptive.json \
  --device cuda
```

The output `threshold_indices_by_image` is consumed by the optional few-shot
shared rank-offset calibrator. A global scalar threshold and RC-Direct are
required ablations, not the proposed full-curve adaptive method.

Run the label-free count-all upper-bound baseline as a strong sanity check:

```bash
$PY -m evaluation.count_all_baseline \
  --warmup-score-dir outputs/score_maps/target_warmup_native \
  --future-score-dir outputs/score_maps/target_future_native \
  --formal \
  --threshold-grid artifacts/threshold_grid.npy \
  --pixel-budget 1e-6 --component-budget 1.0 \
  --output outputs/zero_label/count_all.json
```

`--formal` binds both inputs to their version-3 record hashes and mask-alignment
audit. The warm-up manifest may declare either labeled or mask-free export, but
threshold selection still loads only probabilities; a supplied future split
must be a fully verified labeled export.

### 5. Optional few-shot CRC: build calibration curves, calibrate, and audit

This stage is separate from static cross-fit. It is run only when the dataset
provides labelled calibration images that are disjoint from final-test images;
the same official test images cannot serve both roles. Build the count/loss
archives independently with the frozen native-resolution detector:

```bash
$PY -m certification.build_calibration_losses \
  --score-dir outputs/score_maps/target_calibration_native \
  --threshold-grid artifacts/threshold_grid.npy \
  --pixel-budget 1e-6 --component-budget 1.0 \
  --loss-mode budget_violation \
  --output outputs/certification/calibration_curves.npz

$PY -m certification.build_calibration_losses \
  --score-dir outputs/score_maps/target_test_native \
  --threshold-grid artifacts/threshold_grid.npy \
  --pixel-budget 1e-6 --component-budget 1.0 \
  --loss-mode budget_violation \
  --output outputs/certification/test_curves.npz

$PY -m certification.calibrate_target_offset \
  --calibration-curves outputs/certification/calibration_curves.npz \
  --test-curves outputs/certification/test_curves.npz \
  --zero-result outputs/zero_label/selection_adaptive.json \
  --alpha 0.1 \
  --output outputs/certification/selection.json

$PY -m certification.evaluate_certified_mode \
  --selection-result outputs/certification/selection.json \
  --test-curves outputs/certification/test_curves.npz \
  --output outputs/certification/test_audit.json
```

The finite-sample correction cannot attain a bound below `1/(m+1)`.
Consequently `m=5, alpha=0.1` is infeasible (`1/6 > 0.1`) and must return a
reject result rather than a certificate; `alpha=0.1` requires at least nine
calibration images even when empirical bounded loss is zero. A successful
few-shot result is conditional on the recorded exchangeability,
pre-specification, monotone-loss, frozen-detector, and split-separation
assumptions; it does not certify unbounded raw false-alarm counts.
In the default binary loss mode the controlled expectation is the marginal
probability of violating either the pixel budget or the conservative component
suffix-envelope budget. It therefore implies a conservative raw-component
`JointBSR >= 1 - alpha` statement under those assumptions; it is not an exact
equivalence. The expectation is also marginal over calibration-set randomness
and a fresh exchangeable causal block, not a conditional high-probability
certificate for one realised calibration set. The optional `risk_ratio` mode
controls only a bounded surrogate and must not be reported as a BSR guarantee.
The independent audit reports overall and active-only TP/GT/Pd/JointBSR,
coverage, and reject rate, so abstention cannot silently improve utility.

## Training
The legacy-compatible and separate entrypoints are both executable:

```bash
"${PYTHON_BIN:-python3}" main.py \
  --dataset-dir datasets/IRSTD-1K --batch-size 4 --epochs 400 \
  --lr 0.05 --mode train

"${PYTHON_BIN:-python3}" train.py \
  --dataset-dir datasets/NUDT-SIRST --batch-size 4 --epochs 400 --lr 0.05
```

Training checkpoints and best weights are saved under `repro_runs/` by default.

New weight files include the epoch and the head used at that epoch. New full
checkpoints contain model/optimizer state and Python-native metrics. Resume only
from a checkpoint you trust: complete checkpoint loading is explicitly
unrestricted for compatibility with PyTorch 2.6+, while inference weights use
the restricted weights-only loader.

## Optional multi-source Tail-CVaR detector ablation

The risk-sensitive training entrypoint draws the same number of samples from
every source domain on each step. SLS remains active on the final prediction and
all auxiliary scales; local-background-peak Tail-CVaR and hard-target Miss-CVaR
are applied to the final prediction.

```bash
"${PYTHON_BIN:-python3}" -m scripts.train_multisource_tail \
  --source-dirs \
    datasets/NUAA-SIRST \
    datasets/NUDT-SIRST \
    datasets/IRSTD-1K \
  --batch-per-domain 2 \
  --epochs 400 \
  --lr 0.05 \
  --lambda-tail 0.1 \
  --lambda-miss 0.1 \
  --tail-q 0.01 \
  --miss-q 0.2 \
  --tail-gamma 10 \
  --save-dir repro_runs/rc_tail
```

Each run writes `config.json`, per-epoch `metrics.jsonl`, explicit fixed-last
`last.pkl`, diagnostic `best_train_loss.pkl`, selected `weight.pkl`, and the
latest complete `checkpoint.pkl`. The full checkpoint includes parent RNG and
per-domain sampler state for epoch-boundary resume. Metrics include the SLS,
tail and miss terms plus every source-domain tail risk and the actual
single-head/multi-scale head state.

## Testing
You can test the model with the legacy-compatible entrypoint:

```bash
"${PYTHON_BIN:-python3}" main.py \
  --dataset-dir datasets/IRSTD-1K --batch-size 4 --mode test \
  --weight-path repro_runs/<RUN_ID>/weight.pkl
```

Or use the separate testing entrypoint:

```bash
"${PYTHON_BIN:-python3}" test.py \
  --dataset-dir datasets/IRSTD-1K \
  --weight-path repro_runs/MSHNet-YYYY-MM-DD-HH-MM-SS/weight.pkl
```

The dataset loader supports both the original `trainval.txt`/`test.txt` layout and the local `img_idx/train_*.txt`/`img_idx/test_*.txt` layout.

`--inference-warm-flag auto` is the default: it reads metadata from new weight
files and treats legacy raw state dicts as fully trained multi-scale weights.
For an old epoch-0 raw smoke weight, select the single head explicitly:

```bash
"${PYTHON_BIN:-python3}" test.py \
  --dataset-dir datasets/NUDT-SIRST \
  --weight-path repro_runs/smoke/MSHNet-2026-07-05-16-52-46/weight.pkl \
  --inference-warm-flag false \
  --num-workers 0
```

Test mode prints the actual head and writes `test_manifest.json` beside the
weight by default. Use `--test-manifest none` to disable it or pass an explicit
path.

## Tests

```bash
"${PYTHON_BIN:-python3}" -m pytest -q tests
```

Historical baseline status and known incomplete runs are recorded in
[`baseline_results.md`](baseline_results.md).

## Upstream visual examples

This image is inherited from the upstream MSHNet repository.

![](assert/visual_result.png)

## Upstream-reported quantitative results

These values are retained for upstream reference and are not RC-IRSTD-v2
cross-domain results.

| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 67.16 | 93.88 | 15.03 | [IRSTD-1k_weights](https://drive.google.com/file/d/1q3zfzJRczodGQb0dZ3y3KmLn0zz4F8ra/view?usp=drive_link) |
| NUDT-SIRST | 80.55 | 97.99 | 11.77 | [NUDT-SIRST_weights](https://drive.google.com/file/d/1uczanUIHePZqJA79RZu25fv9FNSHSDQZ/view?usp=drive_link) |


## Citation
**Please kindly cite the papers if this code is useful and helpful for your research.**

    @inproceedings{liu2024infrared,
      title={Infrared Small Target Detection with Scale and Location Sensitivity},
      author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
      booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition},
      year={2024}
    }
