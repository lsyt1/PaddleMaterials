import random
import numpy as np
import paddle
from typing import Dict, List, Optional

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

def compute_mae(pred: paddle.Tensor, target: paddle.Tensor) -> float:
    return float(paddle.mean(paddle.abs(pred - target)).numpy())

def compute_rmse(pred: paddle.Tensor, target: paddle.Tensor) -> float:
    return float(paddle.sqrt(paddle.mean((pred - target)**2)).numpy())

def compute_atomic_energies(atomic_numbers: List[int]) -> Dict[int, float]:
    e0s = {}
    for z in atomic_numbers:
        # 简单近似，可替换为实际 DFT 值
        e0s[z] = -500.0 * np.sqrt(z)
    return e0s

def save_checkpoint(model, optimizer, epoch, loss, path, scheduler=None):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    paddle.save(ckpt, path)

def load_checkpoint(model, path, optimizer=None, scheduler=None):
    ckpt = paddle.load(path)
    model.set_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.set_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.set_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0)
