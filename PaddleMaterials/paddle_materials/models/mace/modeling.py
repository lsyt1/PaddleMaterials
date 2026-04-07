import math
import paddle
import paddle.nn as nn
from typing import List, Dict, Optional, Tuple
from .configuration import MACEConfig
from .utils import compute_atomic_energies

class RadialBasis(nn.Layer):
    def __init__(self, r_max: float, num_basis: int):
        super().__init__()
        self.r_max = r_max
        self.num_basis = num_basis
        freqs = paddle.arange(1, num_basis + 1, dtype='float32') * math.pi / r_max
        # 使用 Assign 初始化器将 freqs 张量赋值给参数
        self.freqs = self.create_parameter(
            shape=[num_basis],
            dtype='float32',
            default_initializer=nn.initializer.Assign(freqs)
        )
    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        r = r.unsqueeze(-1)
        basis = paddle.sin(self.freqs * r) / (r + 1e-8)
        return paddle.where(r < 1e-8, paddle.ones_like(basis), basis)

class CutoffFunction(nn.Layer):
    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = r_max
    def forward(self, r: paddle.Tensor) -> paddle.Tensor:
        x = r / self.r_max
        mask = (x < 1.0) & (x > 0.0)
        f = 1 - 6*x**5 + 15*x**4 - 10*x**3
        return paddle.where(mask, f, paddle.zeros_like(r))

class SphericalHarmonics(nn.Layer):
    def __init__(self, max_ell: int):
        super().__init__()
        self.max_ell = max_ell
    def forward(self, direction: paddle.Tensor) -> List[paddle.Tensor]:
        x, y, z = direction[:,0], direction[:,1], direction[:,2]
        r = paddle.sqrt(x*x + y*y + z*z) + 1e-8
        x, y, z = x/r, y/r, z/r
        out = []
        Y0 = paddle.ones_like(x) * math.sqrt(1.0/(4*math.pi))
        out.append(Y0.unsqueeze(-1))
        if self.max_ell >= 1:
            Y1_1 = math.sqrt(3.0/(4*math.pi)) * y
            Y1_0 = math.sqrt(3.0/(4*math.pi)) * z
            Y1_1p = math.sqrt(3.0/(4*math.pi)) * x
            out.append(paddle.stack([Y1_1, Y1_0, Y1_1p], axis=-1))
        if self.max_ell >= 2:
            c = math.sqrt(15.0/(4*math.pi))
            Y2_2 = c * x * y
            Y2_1 = c * y * z
            Y2_0 = c * (2*z*z - x*x - y*y) / 2
            Y2_1p = c * x * z
            Y2_2p = c * (x*x - y*y) / 2
            out.append(paddle.stack([Y2_2, Y2_1, Y2_0, Y2_1p, Y2_2p], axis=-1))
        if self.max_ell >= 3:
            c1 = math.sqrt(35.0/(4*math.pi))
            c2 = math.sqrt(105.0/(2*math.pi))
            c3 = math.sqrt(21.0/(4*math.pi))
            Y3_3 = c1 * (x*x - y*y) * y
            Y3_2 = c2 * x * y * z
            Y3_1 = c3 * y * (4*z*z - x*x - y*y) / 4
            Y3_0 = c1 * z * (2*z*z - 3*x*x - 3*y*y) / 2
            Y3_1p = c3 * x * (4*z*z - x*x - y*y) / 4
            Y3_2p = c2 * (x*x - y*y) * z / 2
            Y3_3p = c1 * (x*x*x - 3*x*y*y)
            out.append(paddle.stack([Y3_3, Y3_2, Y3_1, Y3_0, Y3_1p, Y3_2p, Y3_3p], axis=-1))
        return out

class InteractionBlock(nn.Layer):
    def __init__(self, config: MACEConfig):
        super().__init__()
        self.config = config
        self.radial_basis = RadialBasis(config.r_max, config.num_radial_basis)
        self.cutoff = CutoffFunction(config.r_max)
        self.edge_mlp = nn.Sequential(
            nn.Linear(config.num_radial_basis, config.num_channels),
            nn.Swish(),
            nn.Linear(config.num_channels, config.num_channels)
        )
        self.node_update = nn.Linear(config.num_channels, config.num_channels)
        self.gate = nn.Swish()
    def forward(self, node_feats: paddle.Tensor, edge_index: paddle.Tensor,
                edge_dist: paddle.Tensor, edge_vec: paddle.Tensor,
                batch=None) -> paddle.Tensor:
        radial = self.radial_basis(edge_dist)
        cutoff = self.cutoff(edge_dist)
        edge_feats = self.edge_mlp(radial) * cutoff.unsqueeze(-1)
        dst = edge_index[0]
        agg = paddle.zeros_like(node_feats)
        agg = paddle.scatter_add(agg, dst, edge_feats, overwrite=False)
        new = self.node_update(node_feats) + agg
        return self.gate(new)

class ReadoutBlock(nn.Layer):
    def __init__(self, config: MACEConfig, atomic_energies: Dict[int, float]):
        super().__init__()
        self.config = config
        self.atomic_energies = atomic_energies
        self.energy_head = nn.Sequential(
            nn.Linear(config.num_channels, config.num_channels),
            nn.Swish(),
            nn.Linear(config.num_channels, 1)
        )
    def forward(self, node_feats: paddle.Tensor, atomic_numbers: paddle.Tensor,
                positions: paddle.Tensor, batch: Optional[paddle.Tensor] = None
               ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        e0_list = [self.atomic_energies.get(int(z.item()), 0.0) for z in atomic_numbers]
        e0 = paddle.to_tensor(e0_list, dtype='float32')
        per_atom = self.energy_head(node_feats).squeeze(-1) + e0
        if batch is None:
            energy = paddle.sum(per_atom)
        else:
            max_batch = paddle.max(batch)
            energy = paddle.zeros([max_batch + 1], dtype='float32')
            energy = paddle.scatter_add(energy, batch, per_atom)
        forces = None
        if paddle.is_grad_enabled() and not positions.stop_gradient:
            forces = paddle.grad(energy, positions, create_graph=True)[0]
            forces = -forces
        return energy, forces

class MACEModel(nn.Layer):
    def __init__(self, config: MACEConfig, atomic_numbers: Optional[List[int]] = None):
        super().__init__()
        self.config = config
        if atomic_numbers is None:
            atomic_numbers = list(range(1, 119))
        self.atomic_numbers = atomic_numbers
        self.num_atom_types = len(atomic_numbers)
        self.atom_embedding = nn.Embedding(self.num_atom_types, config.num_channels)
        self.interaction_blocks = nn.LayerList([
            InteractionBlock(config) for _ in range(config.num_blocks)
        ])
        atomic_energies = compute_atomic_energies(atomic_numbers)
        self.readout = ReadoutBlock(config, atomic_energies)
        self._init_weights()

    def _init_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Linear):
                nn.initializer.XavierUniform()(layer.weight)
                if layer.bias is not None:
                    nn.initializer.Constant(0.0)(layer.bias)
            elif isinstance(layer, nn.Embedding):
                nn.initializer.Normal(mean=0.0, std=0.02)(layer.weight)

    def _get_atom_type_idx(self, atomic_numbers: paddle.Tensor) -> paddle.Tensor:
        idx = paddle.zeros_like(atomic_numbers, dtype='int64')
        for i, z in enumerate(self.atomic_numbers):
            idx = paddle.where(atomic_numbers == z, paddle.full_like(idx, i), idx)
        return idx

    def forward(self, atomic_numbers: paddle.Tensor, positions: paddle.Tensor,
                edge_index: paddle.Tensor, edge_dist: paddle.Tensor, edge_vec: paddle.Tensor,
                batch: Optional[paddle.Tensor] = None) -> Dict[str, paddle.Tensor]:
        atom_idx = self._get_atom_type_idx(atomic_numbers)
        x = self.atom_embedding(atom_idx)
        for block in self.interaction_blocks:
            x = block(x, edge_index, edge_dist, edge_vec, batch)
        energy, forces = self.readout(x, atomic_numbers, positions, batch)
        return {'energy': energy, 'forces': forces}
