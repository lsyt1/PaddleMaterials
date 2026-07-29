# LiFlow

[Flow Matching for Accelerated Simulation of Atomic Transport in Crystalline Materials](https://www.nature.com/articles/s42256-025-01125-4) (Nature Machine Intelligence, 2025)

## Abstract

LiFlow is a generative framework for accelerating molecular dynamics simulations of crystalline materials. It learns distributions of atomic displacements from reference trajectories and uses a propagator and a corrector to advance atomic configurations over time. The method targets long-timescale atomic transport while preserving the structural statistics observed in molecular dynamics trajectories.

## Datasets

| Dataset | Description | Download |
|---------|-------------|----------|
| Universal MLIP / LGPS | Official LiFlow trajectory data | [Zenodo 14889658](https://zenodo.org/records/14889658) ([DOI](https://doi.org/10.5281/zenodo.14889658)) |
| LPS | Available from the original authors upon request | — |

Download every `data.tar.gz.part.*` file from Zenodo, then merge and extract into `data/liflow`:

```bash
cat data.tar.gz.part.* > data.tar.gz
mkdir -p data/liflow
tar -xvf data.tar.gz -C data/liflow
```

Each dataset directory contains:

| File | Description |
|------|-------------|
| `element_index.npy` | Atomic species indices with shape `[n_elements]` |
| `atomic_numbers.npy` | Atomic numbers indexed by structure name; each value has shape `[n_atoms]` |
| `lattice.npy` | Lattice matrices indexed by structure name; each value has shape `[3, 3]` |
| `positions_{temp}K.npz` | Position trajectories indexed by structure name; each value has shape `[n_frames, n_atoms, 3]` |
| `{train,test}_{temp}K.csv` | Training and testing trajectory indices |

CSV columns include `name`, `temp`, `t_start`, `t_end`, `comp`, `msd_t_Li`, `msd_t_frame`, `prior_Li`, and `prior_frame`. See the [original LiFlow repository](https://github.com/learningmatter-mit/liflow) for definitions and dataset-specific preparation details.

## Models

LiFlow uses two flow-matching models:

- The **propagator** predicts atomic displacements over a coarse simulation interval.
- The **corrector** refines propagated structures using a noise-conditioned flow.

Both models operate on periodic crystal structures and condition the predicted atomic velocity field on atomic species, positions, simulation time, and temperature.

## Results

| Model | Dataset | Metric | Paddle result | Config | Checkpoint / Log |
|-------|---------|--------|---------------|--------|------------------|
| LiFlow | Universal / LGPS | Pending evaluation | Pending | [liflow_universal](configs/liflow_universal.yaml) | [checkpoint](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/molecular_dynamics_integrator/liflow/liflow_universal.zip) |

### Training

```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" \
  molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml

# single-gpu training
python molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml
```

### Validation

```bash
python molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml \
  Global.do_eval=True Global.do_train=False Global.do_test=False \
  Trainer.pretrained_model_path='your_model.pdparams'
```

### Testing

```bash
python molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml \
  Global.do_test=True Global.do_train=False Global.do_eval=False \
  Trainer.pretrained_model_path='your_model.pdparams'
```

### Prediction

Mode 1: local config and checkpoint

```bash
python molecular_dynamics_integrator/predict.py \
  --config_path molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml \
  --checkpoint_path your_model.pdparams \
  --data_path data/liflow \
  --index_file test_800K.csv
```

Mode 2: registered pretrained model (auto-download from BCE)

```bash
python molecular_dynamics_integrator/predict.py \
  --model_name liflow_universal \
  --data_path data/liflow \
  --index_file test_800K.csv
```

## Citation

```bibtex
@article{nam2025flow,
  title={Flow Matching for Accelerated Simulation of Atomic Transport in Crystalline Materials},
  author={Juno Nam and Sulin Liu and Gavin Winter and KyuJung Jun and Soojung Yang and Rafael G{\'o}mez-Bombarelli},
  journal={Nature Machine Intelligence},
  year={2025},
  doi={10.1038/s42256-025-01125-4},
}
```
