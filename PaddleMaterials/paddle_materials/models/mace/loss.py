import paddle
import paddle.nn as nn

# MACE 损失：能量损失 + 力损失（加权）
class MACELoss(nn.Layer):
    def __init__(self, energy_weight=1.0, force_weight=100.0):
        super().__init__()
        self.energy_w = energy_weight
        self.force_w = force_weight

    def forward(self, pred_energy, target_energy, pred_force, target_force):
        loss_e = nn.functional.mse_loss(pred_energy, target_energy)
        loss_f = nn.functional.mse_loss(pred_force, target_force)
        return self.energy_w * loss_e + self.force_w * loss_f