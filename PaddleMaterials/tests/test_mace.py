import os
import sys
import pytest
import numpy as np
import paddle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paddle_materials.models.mace import MACEModel, MACEConfig, MACEDataset, MACELoss
from paddle_materials.models.mace.utils import set_seed, compute_mae

@pytest.fixture
def config():
    c = MACEConfig()
    c.num_channels = 32
    c.num_blocks = 1
    c.num_radial_basis = 4
    c.max_ell = 2
    return c

@pytest.fixture
def model(config):
    set_seed(42)
    return MACEModel(config, atomic_numbers=list(range(1,10)))

@pytest.fixture
def sample_batch(config):
    set_seed(42)
    n_atoms = 10
    n_edges = 30
    return {
        'atomic_numbers': paddle.randint(1,10, [n_atoms]),
        'positions': paddle.randn([n_atoms,3]),
        'edge_index': paddle.randint(0, n_atoms, [2, n_edges]),
        'edge_dist': paddle.rand([n_edges]) * config.r_max,
        'edge_vec': paddle.randn([n_edges,3]),
        'batch': None
    }

def test_model_creation(model):
    assert model is not None

def test_forward_shape(model, sample_batch):
    out = model(**sample_batch)
    assert 'energy' in out
    assert 'forces' in out
    assert out['forces'] is not None

def test_forward_stability(model, sample_batch):
    set_seed(42)
    out1 = model(**sample_batch)
    out2 = model(**sample_batch)
    diff = paddle.max(paddle.abs(out1['energy'] - out2['energy'])).numpy()
    assert diff < 1e-6

def test_training_step(model, sample_batch):
    opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
    out = model(**sample_batch)
    target_energy = paddle.randn([1]) if out['energy'].ndim==0 else paddle.randn([out['energy'].shape[0]])
    target_forces = paddle.randn(sample_batch['positions'].shape)
    loss_fn = MACELoss(model.config)
    loss = loss_fn(out['energy'], out['forces'], target_energy, target_forces)['total_loss']
    loss.backward()
    opt.step()
    opt.clear_grad()
    assert loss.item() > 0

def test_loss_function(config):
    loss_fn = MACELoss(config)
    pred_e = paddle.randn([4])
    pred_f = paddle.randn([10,3])
    target_e = paddle.randn([4])
    target_f = paddle.randn([10,3])
    loss_dict = loss_fn(pred_e, pred_f, target_e, target_f)
    assert 'total_loss' in loss_dict

if __name__ == '__main__':
    pytest.main([__file__, '-v'])