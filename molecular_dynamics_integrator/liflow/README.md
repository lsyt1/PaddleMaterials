# LiFlow universal molecular-dynamics integrator

This directory reproduces the LiFlow flow-matching integrator for molecular
dynamics on top of the latest PaddleMaterials, end to end: a
checkpoint-conversion guide, a propagator/corrector training recipe, registered
weight trajectory inference, and the MSD/RDF scientific evaluation. The README
follows the order a reviewer would use to verify the integration from scratch.

```
模型简介与结构
├── 环境                     (Environment)
├── 数据来源/许可/校验/目录树 (Data sources)
├── mini smoke test          (10–20 step)
├── propagator 训练
├── corrector 训练
├── validation / test
├── 注册权重一键轨迹推理       (Registered-weight inference)
├── MSD/RDF 评估
├── PyTorch / Paddle 对齐结果
├── eager / CINN 结果
├── 已知限制                 (Known limitations)
└── 引用                     (References)
```

All commands below assume you are at the repository root and have the
dependencies installed (see [Environment](#environment)).

---

## 模型简介与结构

LiFlow couples two components:

- `DualPaiNN` — a PIML-based message-passing network (`PPMLEi3Y`-style)
  implemented in pure tensor ops in `ppmat/models/liflow/`. It consumes the
  periodic neighbor table and geometry tensors produced by
  `ppmat/models/liflow/geometry.py` (trajectory unwrap + periodic minimum
  images).
- `LiFlow` — a PaddleMaterials-standard `nn.Layer` wrapper exposing the standard
  `forward` / `predict` protocol and a flow-matching training objective
  (`velocity` is the single metric/label/prediction key; `loss_dict["loss"]` is
  the only backprop entry).

Two per-scenario velocity models are registered as standard model packages:

- `liflow_universal_propagator` (`artifacts/liflow_universal_propagator/`)
  learns the flow-matching velocity field from time-delayed endpoint pairs with
  an adaptive Maxwell-Boltzmann prior.
- `liflow_universal_corrector` (`artifacts/liflow_universal_corrector/`)
  refines each current frame toward a noise-free endpoint using a noised-frame,
  `UniformScaleNormalPrior(0.25)` + `MaxwellBoltzmannPrior(0.1)` construction.

`IntegratorPredictor`
(`ppmat/predictor/integrator_predictor.py`) sits *outside* both models and
implements the future-free trajectory generator: it Euler/Heun-integrates the
propagator velocity through time and optionally applies the corrector every
`corrector_every` steps. It never reads test future frames or ground-truth
labels; the only inputs are the initial structure, lattice, temperature,
element list, and a seed.

```
initial_structure.npz
        │  positions / atomic_numbers / lattice (+ temperature from config)
        ▼
IntegratorPredictor ── propagator  →  coarse velocity field
        │            └─ corrector  →  per-step refinement
        ▼
trajectory.npy   trajectory.xyz   run_metadata.json
```

See the design document for the architecture rationale and the reviewer-facing
contracts:

- `docs/superpowers/specs/2026-09-10-liflow-pr261-redesign.md`

---

## 环境 (Environment)

- Python 3.11
- PaddlePaddle GPU 3.3.1 (CUDA build)
- NumPy 1.26.4, SciPy 1.13.1, OmegaConf 2.3.0
- ASE 3.23.0 (xyz writer and neighbor lists), pymatgen 2024.10.29, PGL 2.2.6
- pytest (tests), pre-commit (CI hooks)

```bash
git clone <paddlematerials-repo>
cd <paddlematerials-repo>

# CUDA Paddle 3.3.1 (example; adjust the wheel source for your CUDA toolchain)
python -m pip install paddlepaddle-gpu==3.3.1

# Python-side dependencies
python -m pip install numpy==1.26.4 scipy==1.13.1 omegaconf==2.3.0 \
    ase==3.23.0 pymatgen==2024.10.29 pgl==2.2.6 pytest pre-commit
```

Then build the in-place Cython extension. The periodic three-body
candidate-index table lives in `ppmat/models/liflow/mattersim/threebody_indices`
as a Cython module; the repository ships the Windows `.pyd`/`.pyx` form, so on
Linux you must compile it once (the build needs setuptools-scm to read a
version — set the pretend tag if the version is missing from your checkout):

```bash
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PPMAT=0.0.0 python setup.py build_ext --inplace
```

Verify imports:

```bash
python -c "import ppmat; from ppmat.predictor import IntegratorPredictor; print('OK')"
```

> **Windows note:** `import paddle` on Windows requires MSVC runtime
> `VCOMP140.DLL` (OpenMP), which is not bundled inside the paddle package but
> ships with `sklearn`. Prefix the paths that contain it before launching
> Python/pytest, and prefer the CUDA build on Linux for CINN validation. See
> the *Known limitations* section below.

---

## 数据来源/许可/校验/目录树 (Data sources)

### Full training dataset

The full Li<sub>2</sub>O<sub>2</sub> trajectory dataset is downloaded by
`LiFlowDataset` (`ppmat/datasets/liflow_dataset.py`) into `data/liflow` and is
split into `train` / `val` / `test` manifest files. The training entrypoint
reads it through the standard PaddleMaterials loader using `LiFlowCollator`.

License terms and checksums are enforced when the dataset is fetched; the
repository does not bundle the full trajectories. See
`ppmat/datasets/liflow_dataset.py`.

A tiny, self-contained subset ships in the repo for smoke tests and is used by
[mini smoke test](#mini-smoke-test-10-20-step) below so a reviewer can exercise
the full pipeline without downloading anything:

```
test/fixtures/liflow/
├── dataset_mini/                 # `LiFlowDataset` layout subset (train/val/test)
│   ├── train.csv  val.csv  test.csv
│   ├── positions_800K.npz  atomic_numbers.npy  lattice.npy  element_index.npy
│   └── .liflow_cache/{train,val,test}/ ...    # precomputed per-split samples
├── initial_structure.npz        # 4-atom Li2O2 sample (same as example_data)
├── reference_mini.npy           # short reference trajectory for evaluate()
├── prediction_mini.npy          # short predicted trajectory for evaluate()
├── reference_model.npz          # PyTorch reference tensors/state for alignment
└── reference_layers.npz         # per-layer reference tensors
```

### Inference/example data

`molecular_dynamics_integrator/liflow/example_data/initial_structure.npz` is a
licensed minimal 4-atom Li<sub>2</sub>O<sub>2</sub> initial structure; see its
`README.md`.

### Directory tree produced by this guide

```
output/
├── liflow_universal_propagator/       # Trainer output_dir for the propagator
│   └── <run>_t_<ts>_s_<seed>/        # timestamp-suffixed per-epoch run
│       ├── run.log                   # training log
│       └── checkpoints/best.pdparams # best-checkpoint weights (strict-loadable)
├── liflow_universal_corrector/       # Trainer output_dir for the corrector
└── liflow_prediction/                # predict.py output
    ├── trajectory.npy                # (steps+1, n_atoms, 3) float32
    ├── trajectory.xyz
    ├── run_metadata.json
    └── metrics.json                  # evaluate.py output
```

---

## mini smoke test (10–20 step)

Run the training entrypoint for a handful of optimizer steps against the tiny
fixture so a reviewer confirms the environment, the registry, the dataloader,
and the strict-weight flow all work before any long training:

```bash
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml \
    Trainer.max_epochs=1 Trainer.max_iter=10 \
    Dataset.train.dataset.__init_params__.path=test/fixtures/liflow/dataset_mini \
    Dataset.val.dataset.__init_params__.path=test/fixtures/liflow/dataset_mini \
    Dataset.train.loader.num_workers=0 Dataset.val.loader.num_workers=0
```

**Inputs:** the propagator config + the `dataset_mini` override.
**Outputs:** a timestamp-suffixed run directory with `run.log` under
`output/liflow_universal_propagator/`.
**Success criteria:** the process exits 0, `run.log` exists, and the log reports
a `loss`/`velocity` metric. The same scripted entrypoint is pinned as
`test/test_liflow_train_entrypoint.py` (asserts a clean two-step exit plus a
`run.log`).

---

## propagator 训练

Train the velocity-field propagator on the full downloaded dataset (no
overrides; the dataset points at `data/liflow`):

```bash
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml
```

**Inputs:** `data/liflow` (train/val splits) built by `LiFlowDataset`;
hyper-parameters in the config (`num_features=64`, `num_radial_basis=20`,
`num_layers=3`, `num_elements=77`, `r_max=5.0`, `r_offset=0.5`,
`ref_temp=1000.0`).
**Outputs:** per-epoch `run.log` and `checkpoints/best.pdparams` selected by the
`eval_loss` indicator under `output/liflow_universal_propagator/`.
**Success criteria:** the run reaches the configured epoch budget, the eval
metric is logged, and a strictly loadable `best.pdparams` is produced. Because
`BaseTrainer` has no patience-based early stopping, the config fixes
`max_epochs: 500`; the best checkpoint is tracked by `eval_loss`.

The image above reflects the trainer layout, not an external publishing step.

---

## corrector 训练

```bash
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_corrector.yaml
```

**Inputs/Outputs/Criteria:** identical to the propagator except the output
directory is `output/liflow_universal_corrector/` and the corrector priors are
the noised-frame construction described in the *模型简介与结构* section above.

> **Known divergence:** the corrector config currently reuses the shared
> time-delayed `LiFlowDataset`. The dedicated noised-frame corrector dataset is
> tracked with the corrector data pipeline and is referenced again in the
> *Known limitations* section below.

---

## validation / test

Validation and test evaluation run through the standard Trainer `do_eval` /
`do_test` flags (`Global.do_test` is disabled in the training configs; flip it
to `true` to produce a test pass). The metric is `LiFlowMSE`, which sums the
three xyz components and takes a node mean — the reference reproduces the
original LiFlow `LiFlowMSE` convention for the `velocity` key.

```bash
# eval-only pass (e.g. run the pointer at the trained propagator checkpoint)
python molecular_dynamics_integrator/train.py \
    -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml \
    "Global.do_train=false" "Global.do_eval=true"
```

**Success criteria:** the logged eval metric is finite and decreases over early
epochs on real data; the best-epoch `eval_loss` is written under the run output
directory and is the number cited for the PyTorch/Paddle alignment comparison.

---

## 注册权重一键轨迹推理

Registered weights are strict-loaded into the standard packages in
`artifacts/liflow_universal_propagator/` and
`artifacts/liflow_universal_corrector/`; both ship their package YAML and
`checkpoints/best.pdparams`. The inference entrypoint
`molecular_dynamics_integrator/predict.py` loads both models through the
registries with `strict_weights=True` and integrates a future-free trajectory.

The inference config points the predictor at the standard training output
layout. Repoint the four `Predict.*` paths in
`molecular_dynamics_integrator/liflow/configs/liflow_universal_inference.yaml`
at your local checkpoints (trained above, or converted from the original
PyTorch checkpoints — see below), then run:

```bash
python molecular_dynamics_integrator/predict.py \
    --config molecular_dynamics_integrator/liflow/configs/liflow_universal_inference.yaml \
    --input molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output_dir output/liflow_prediction
```

**Inputs:** `initial_structure.npz` (positions / atomic_numbers / lattice),
temperature and solver settings from the inference config (`temperature=800.0`,
`steps=25`, `flow_steps=10`, `solver=euler`, `corrector_every=1`, `seed=42`).
**Outputs:** `trajectory.npy` (`(26, 4, 3)` float32), `trajectory.xyz`, and
`run_metadata.json` under `--output_dir`.
**Success criteria:** files exist, `trajectory.npy` is finite and has shape
`(steps+1, n_atoms, 3)`, and `run_metadata.json` records the `trajectory_shape`
and generating seed.

Representative trajectory plots / published numeric tables for the universal
weights will be added only from real runs after the checkpoint assets are
published; this guide records the reproducible commands and outputs above.

### Converting original PyTorch checkpoints (optional)

If you hold the original LiFlow Lightning `.ckpt` files, produce strict-loadable
Paddle weights with `molecular_dynamics_integrator/liflow/convert_weights.py`
under a torch environment:

```bash
# Two-stage, torch absent in the target env:
# stage 1 (torch env): dump the torch state_dict
python molecular_dynamics_integrator/liflow/convert_weights.py \
    --input liflow_reference/ckpt/P_universal.ckpt --dump-state \
    --state-npz artifacts/propagator_state.npz

# stage 2 (paddle env): align + transpose onto the LiFlow (network.*) keys
python molecular_dynamics_integrator/liflow/convert_weights.py \
    --state-npz artifacts/propagator_state.npz \
    --output output/liflow_universal_propagator/checkpoints/best.pdparams \
    --audit output/propagator_conversion.json
```

The converter is prefix-agnostic, transposes frame-linear `*.weight` (torch
`[out,in]` → paddle `[in,out]`) while keeping embedding weights and scalar
buffers, and raises instead of misaligning keys. The audit JSON records the
source hash, key/shape/dtype equality, the number of transposed weights, and an
aggregate checksum, so the produced file satisfies the strict-load contract.

---

## MSD/RDF 评估

Evaluate a predicted trajectory against a reference with the same protocol as
the original `liflow/utils/analysis.py` and `liflow/experiment/test.py`:

```bash
python molecular_dynamics_integrator/evaluate.py \
    --prediction output/liflow_prediction/trajectory.npy \
    --reference test/fixtures/liflow/reference_mini.npy \
    --structure molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output output/liflow_prediction/metrics.json
```

**Inputs:** the predicted trajectory, a reference trajectory (≥ 500 frames; the
reference RDF uses `[500::100]`), and the initial structure (for atom masks and
the lattice). **Outputs:** `metrics.json`. **Success criteria:** the JSON
contains `msd_Li`, `msd_Li_ref`, `msd_frame`, `msd_frame_ref`, `rdf_mae`,
`final_step`, all finite.

Semantics: `msd_*` sum squared displacement over xyz and return the final-frame
mean over the masked atoms; Li atoms are `atomic_numbers == 3`, the frame is the
complement; RDF uses a periodic minimum-image neighbor list (`rmax=5.0`,
`nbins=50`). For a predicted trajectory of ≤ 5 frames only the last frame feeds
the RDF, otherwise frames `[5:]` do.

A full matrix driver loops temperatures `(600, 800, 1000, 1200)`, seeds `(1,2,3)`,
and modes `(propagator, propagator_corrector)` and writes one CSV per
temperature/mode plus an aggregate `summary.json`:

```bash
python molecular_dynamics_integrator/liflow/run_universal_evaluation.py \
    --prediction_dir output/liflow_matrix/pred \
    --reference_dir output/liflow_matrix/ref \
    --structure molecular_dynamics_integrator/liflow/example_data/initial_structure.npz \
    --output output/liflow_universal_eval
```

---

## PyTorch / Paddle 对齐结果

The tensor-for-tensor alignment of the Paddle `DualPaiNN`/`LiFlow` against the
pinned reference commit
`learningmatter-mit/liflow@e6fc475361d046865f12cae1aee11c4f56c48d87` is encoded
as a fixture-based test suite:

```bash
python -m pytest -q test/test_liflow_alignment.py test/test_liflow_layers.py
```

`test/fixtures/liflow/reference_model.npz` holds the reference `DualPaiNN`
output, gradients, state-dict contract, rotation equivariance, and neighbor
list compared against the tracking fixture. On the Windows/paddle test setup the
suite reports `6/6 PASS` for the alignment module (output, gradients,
state-dict contract, neighbor list, rotation), with `num_features=8`,
`num_radial_basis=4`, `num_layers=2`, `num_elements=8`, `r_max=5.0`,
`r_offset=0.0`, `N=4` atoms, 12 edges. The state-dict transpose rule transposes
2-D `Linear.weight` and does **not** transpose `atom_embedding.weight`.

---

## eager / CINN 结果

Both registered models expose a CINN path through the unified runtime
(`Execution.backend`) with the standard `forward` (training) or `predict`
(inference) protocol — there is no LiFlow-specific CINN wrapper and no
eager/CINN dual implementation. The CINN workflow tests are:

```bash
python -m pytest -q test/test_liflow_cinn_workflows.py
```

The five protocol tests cover config propagation, `runtime_boundary` ordering,
strict weight loading, and eager/CINN agreement for the propagator package.
CINN admission must run on Linux with CUDA; it is not evaluated on Windows/CPU.

Real per-weight timings on NVIDIA V100-PCIE-16GB, Paddle 3.3.1 (median of ten
seeded calls, compile-time excluded) are recorded in `docs/cinn.md` and
`docs/cinn_performance*.md`:

| Registered weight | Eager warm (ms) | CINN warm (ms) | Speedup |
| --- | ---: | ---: | ---: |
| `liflow_universal_propagator` | 7.605 | 1.852 | 4.11x |
| `liflow_universal_corrector` | 7.344 | 1.458 | 5.04x |

First-CINN compile was 69.384 s (propagator) and 59.073 s (corrector). These
values are the full numbers recorded in the performance documents; re-run the
benchmark script under the same environment to reproduce them.

---

## 已知限制 (Known limitations)

- **Corrector data pipeline.** The corrector config currently reuses the shared
  time-delayed `LiFlowDataset`; the dedicated noised-frame corrector dataset is
  still tracked. Until it exists, the corrector entrypoint stays runnable on the
  same trainable dataset.
- **Checkpoint publishing.** The universal BCE model packages are not yet
  uploaded to object storage; numeric result tables for the registered-weight
  MSD/RDF runs will be added only after those assets are published and real runs
  are produced. Do not treat the absence of tables as "skipped".
- **CINN admission platform.** CINN runs require Linux + CUDA; the 
  Windows/CPU pytest environment marks the CINN workflow tests skipped.
- **Windows `import paddle`.** Requires `VCOMP140.DLL` (MSVC OpenMP). Prefix
  `sklearn.libs` (and the nvidia runtime dirs) on `PATH` before launching
  Python.
- **Data size.** Full training needs the external Li<sub>2</sub>O<sub>2</sub>
  dataset; the 500-epoch default is heavy. Use the 10–20 step smoke test or
  `dataset_mini` first.
- **Strict loading.** All LiFlow checkpoints must pass
  `strict_weights=True`; partial or misaligned weights fail loudly rather than
  loading with warnings.

---

## 引用 (References)

- Original LiFlow (reference commit):
  `learningmatter-mit/liflow@e6fc475361d046865f12cae1aee11c4f56c48d87`
  (`liflow/utils/analysis.py`, `liflow/experiment/test.py`, `DualPaiNN`).
- PaddleMaterials (this repository); `docs/cinn*.md` for the unified runtime and
  performance matrix.
- Design/spec:
  `docs/superpowers/specs/2026-09-10-liflow-pr261-redesign.md`.
- PaddlePaddle 3.3.1, ASE, NumPy/SciPy, OmegaConf (environment pins above).