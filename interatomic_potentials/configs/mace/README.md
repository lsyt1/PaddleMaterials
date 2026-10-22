# MACE-MP-0
[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2401.00096)
## Abstract
MACE (Many-body Atomic Cluster Expansion) is a novel machine learning interatomic potential that leverages higher-order equivariant message passing to achieve accurate predictions of atomic energies, forces, and stresses. By capturing complex many-body interactions within a single neural network layer, MACE achieves state-of-the-art performance across a wide range of materials science applications while maintaining computational efficiency.

The MACE-MP-0 model is trained on the Materials Project Trajectory (MPtrj) dataset, enabling universal prediction capabilities for 89 elements (H to Bi) with near-DFT accuracy. The key innovation of MACE lies in its higher-order equivariance, which allows the model to capture 3-body and 4-body interactions simultaneously through symmetric tensor contractions, eliminating the need for explicit many-body terms.

**Model Architecture Overview:**
1. **Input Layer**: Atomic numbers and 3D positions are encoded as node features
2. **Radial Basis Functions**: Interatomic distances are expanded using Bessel/Gaussian basis functions
3. **Spherical Harmonics**: Angular information is encoded using spherical harmonics
4. **Equivariant Message Passing**: Higher-order (3-body and 4-body) messages are computed through symmetric tensor contractions
5. **Interaction Blocks**: Two message passing layers with residual connections
6. **Readout Layer**: Atomic energies are summed to obtain total system energy, forces are computed via automatic differentiation
## Datasets
MACE-MP-0 is trained on the Materials Project Trajectory (MPtrj) dataset, which provides comprehensive coverage of inorganic crystalline materials.

- **MPtrj_2022.9_full**:
    The Materials Project Trajectory Dataset is the primary training dataset for MACE-MP-0. It contains density functional theory (DFT) calculations from the Materials Project database.
    - 145,923 unique compounds
    - 1,580,395 crystal structures
    - Calculations performed at GGA/GGA+U level of theory
    
    Corresponding labels:
    - Total energies
    - Atomic forces
    - Stresses
    
    | Dataset | Train | Val | Test |
    | :--- | :---: | :---: | :---: |
    | MPtrj_2022.9_full | 116738 | 14592 | 14593 |

## Models
MACE constructs an atomistic graph where:
* **Nodes**: Represent atoms with element-specific embeddings
* **Edges**: Encode interatomic distances using radial basis functions
* **Higher-order messages**: Capture 3-body and 4-body interactions simultaneously through equivariant transformations

### Mathematical Formulation
Radial basis functions expand interatomic distances:
$$
\phi_n(r) = \sqrt{\frac{2}{r_c}} \sin\left(\frac{n\pi r}{r_c}\right) / r
$$

Energy prediction:
$$
E_{\text{tot}} = \sum_i E_i
$$

Forces are computed as energy gradients:
$$
\mathbf{F}_i = -\frac{\partial E_{\text{tot}}}{\partial \mathbf{r}_i}
$$

Stresses are derived from the energy–strain relation:
$$
\boldsymbol{\sigma} = \frac{1}{V} \frac{\partial E_{\text{tot}}}{\partial \boldsymbol{\varepsilon}}
$$

## Results

The numbers below come from three different sources and are reported separately.
They must not be compared across blocks.

### A. Original paper results (reference only)

As reported by the MACE authors for MACE-MP-0 medium trained on the full MPtrj
training set. These numbers were **not** produced by this Paddle implementation;
they are listed only as a reference point.

| Model | Dataset | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE (GPa) | GPUs | Training time |
| --- | --- | --- | --- | --- | --- | --- |
| mace_mp0_medium | MPtrj_2022.9_full | 23 | 33 | 0.91 | as reported in the paper | as reported in the paper |

### B. Paddle inference with weights converted from the official PyTorch checkpoint

The checkpoint shipped in `mace_mp0_medium.zip` is obtained by converting the
official PyTorch MACE-MP-0 medium weights into Paddle format. No Paddle training
was involved, so a training time does not apply to this block.

Evaluation protocol:

- set: 754 frames / 60 materials sampled from the MPtrj validation split
  (`uncorrected_total_energy / natoms` for energy, `-0.1 * kBar` for stress,
  forces used as stored);
- energy MAE is reported per atom; stress MAE uses the 6 independent components
  of the symmetric 3x3 tensor.

| Model | Dataset | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE (GPa) | GPUs | Training time |
| --- | --- | --- | --- | --- | --- | --- |
| mace_mp0_medium (converted) | 754 frames / 60 materials (MPtrj val subset) | 17.0 (atom-weighted 12.9) | 61.4 (RMSE 115.2) | 0.42 (6 components, Pearson r = 0.957; 0.31 over all 9 entries) | 1x Tesla V100-PCIE-16GB | n/a (inference only, 0.071 s/frame, 7.8 GB peak) |

This block only characterizes the converted checkpoint on the protocol above.
The subset is a random 60-material sample rather than the official validation
split, and the per-frame error is long-tailed: per-frame force MAE is 44.8 meV/Å
at the median, 122.7 at P90, 322.2 at P99 and 598.3 at the maximum, and the
worst 5% of the frames carry 20.6% of the total force error (36.9% of the stress
error). These values are therefore not directly comparable with the paper
numbers in block A.

The same 754 frames were also evaluated with the official PyTorch MACE
implementation and its original checkpoint
(`_final_check/logs/v19_torch_vs_paddle_all_fix.json`). The two implementations
agree to 1.5e-05 eV/Å on forces, 1.6e-05 eV/atom on energies and 6.6e-05 GPa on
stresses, and reproduce the metrics in the table to every printed digit, so the
numbers above are a property of the converted weights, not of the Paddle port.

### C. Weights trained with Paddle

Results of two short Paddle runs: Stage 0 (300 steps, 368-frame train subset) and
Stage 1 (100 steps, 4,021-frame train subset, two learning rates). Both start
from the converted checkpoint of block B, so they are **fine-tuning from
converted PyTorch weights**, not training from random initialisation. Every
number below is measured, not estimated or carried over from blocks A or B.
Evaluation protocol is identical to block B: the same fixed 754-frame /
60-material MPtrj validation subset, with no test split used for selection.

| Model | Dataset (frames) | Steps | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE (GPa) | GPUs | Peak memory | Training time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mace_mp0_medium (Paddle-trained, Stage 0) | 368 frames / 60 materials (MPtrj train subset) | 300 | 19.9 | 62.7 | 0.44 (6 components; 0.32 over all 9 entries) | 1x Tesla V100-PCIE-16GB | 11,624 MB | 138 s compute (0.46 s/step), 281 s wall including in-loop evaluation |
| mace_mp0_medium (Paddle fine-tuned, Stage 1-A) | 4,021 frames / 938 materials (MPtrj train subset) | 100 (lr 1e-6) | **17.01** | **61.42** | 0.42 (6 components; 0.311 over all 9 entries) | 1x Tesla V100-PCIE-16GB | 8,230 MB | 46.4 s compute (0.46 s/step), 259 s wall including in-loop evaluation |
| mace_mp0_medium (Paddle fine-tuned, Stage 1-B) | 4,021 frames / 938 materials (MPtrj train subset) | 100 (lr 3e-6) | 16.98 | 61.41 | 0.42 (6 components; 0.312 over all 9 entries) | 1x Tesla V100-PCIE-16GB | 8,230 MB | 63.7 s compute (0.64 s/step), 211 s wall including in-loop evaluation |

**Stage 0 is a negative result and is reported as such.** Starting from the
converted checkpoint (block B), 300 Adam steps at a constant learning rate of
3e-5 with the model's own SmoothL1 loss (E:F:S = 1:1:0.1, gradient clip 10,
seed 42) updated all 29 trainable tensors (max absolute change 3.1e-3) and
produced checkpoints that differ from the initial weights, so Paddle training
is genuinely running. However, all three validation metrics degraded
monotonically on the fixed 754-frame subset:

| checkpoint | steps | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE, 9 entries (GPa) |
| --- | --- | --- | --- | --- |
| converted (block B) | 0 | 17.03 | 61.43 | 0.3106 |
| step 100 | 100 | 18.95 | 62.92 | 0.3182 |
| step 200 | 200 | 19.00 | 62.40 | 0.3165 |
| step 300 | 300 | 19.91 | 62.74 | 0.3228 |

The best validation checkpoint is therefore step 0, i.e. the converted weights
themselves; no Stage 0 checkpoint beats block B, and none of them are shipped
in `mace_mp0_medium.zip`.

**Stage 1 fixes most of that degradation.** It uses a 4,021-frame subset drawn
from the official train split by `md5(mp_id)` stratified sampling (938
materials, 31.8 atoms/frame against 31.1 for the full train split, so the
atom-count histogram now matches the full split to within 1 percentage point;
material overlap with our val and test subsets is 0), the same converted
initialisation, the same seed and the same batch order for both candidates.

| candidate | steps | Energy MAE (meV/atom) | Force MAE (meV/Å) | Stress MAE, 9 entries (GPa) | Stress MAE, 6 components (GPa) | relative to block B (E / F / S) |
| --- | --- | --- | --- | --- | --- | --- |
| converted (block B) | 0 | 17.026 | 61.427 | 0.3106 | 0.4225 | — |
| Stage 1-A, lr 1e-6 | 50 | 17.017 | 61.420 | 0.3108 | 0.4228 | −0.05% / −0.01% / +0.08% |
| Stage 1-A, lr 1e-6 | 100 | **17.006** | **61.420** | 0.3110 | 0.4231 | −0.12% / −0.01% / +0.13% |
| Stage 1-B, lr 3e-6 | 50 | 17.004 | 61.416 | 0.3114 | 0.4237 | −0.13% / −0.02% / +0.25% |
| Stage 1-B, lr 3e-6 | 100 | 16.978 | 61.409 | 0.3119 | 0.4244 | −0.29% / −0.03% / +0.41% |

Stage 1-A step 100 is the only checkpoint whose combined E/F/S ratio against
block B is below 1.0 (2.99993, where block B is 3.00000). It is a genuinely
optimised Paddle checkpoint: all 29 trainable tensors changed
(`max |Δ| = 5.0e-5`, SHA256 differs from the initial weights), and it reloads
strictly (63/63 tensors, no missing or unexpected keys, predictions reproduced
to 1.9e-6 eV/Å on forces). Stage 1-B selected step 0 by the same pre-registered
rule because its stress error grew faster than its energy/force error fell.

Both Stage 1 runs are small: energy improves by 0.1%, force by 0.01%, and
stress is still 0.13% worse than block B, so this is **not** a reproduction of
the paper accuracy in block A. No Stage 1 checkpoint is designated as the final
model, and it is not shipped in the zip.

**Note**: the suite checkpoint zip (`mace_mp0_medium.zip`) currently contains the
converted checkpoint of block B together with the config yaml and
`checkpoints/best.pdparams`, so that `predict.py --model_name=mace_mp0_medium`
works out of the box. It has **not** been replaced by any Stage 0 or Stage 1
checkpoint: none of them is better than block B on all three metrics.

### Training
```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
```

### Validation
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Prediction
```bash
# Mode 1: Use pretrained model
python interatomic_potentials/predict.py --model_name='mace_mp0_medium' --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/'
# Mode 2: Use custom checkpoint
python interatomic_potentials/predict.py --config_path='interatomic_potentials/configs/mace/mace_mp0_medium.yaml' --checkpoint_path="your checkpoint path(*.pdparams)" --input_format=cif --input_path='./interatomic_potentials/example_data/cifs/'
```

## Citation
```
@article{batatia2024mace,
  title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={arXiv preprint arXiv:2401.00096},
  year={2024}
}
```
