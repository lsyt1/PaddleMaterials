# ===================================================================
# 文件: paddle_materials/models/mace/configuration.py
# 功能: MACE 模型的配置类，管理所有超参数
# 参考: https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py
# ===================================================================

from typing import Dict, Any, Optional, List


class MACEConfig:
    """
    MACE 模型的配置类。
    
    超参数说明：
    - r_max: 原子间相互作用的截断半径 (Å)，默认 5.0
    - num_radial_basis: 径向基函数数量，默认 8
    - num_cutoff_basis: 截断基函数数量，默认 5
    - max_ell: 球谐函数的最大角动量阶数，默认 3
    - correlation: ACE 关联阶数，默认 3
    - hidden_irreps: 隐藏层等变不可约表示，如 "128x0e + 64x1o"
    - num_channels: 通道数，默认 128
    - num_blocks: 消息传递块数，默认 2
    - energy_weight: 能量损失权重，默认 1.0
    - force_weight: 力损失权重，默认 100.0
    - stress_weight: 应力损失权重，默认 0.01
    - learning_rate: 学习率，默认 1e-3
    - weight_decay: 权重衰减，默认 1e-5
    - batch_size: 批大小，默认 32
    - max_num_epochs: 最大训练轮数，默认 100
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # -------------------- 基础物理参数 --------------------
        self.r_max: float = 5.0              # 截断半径 (Å)
        self.num_radial_basis: int = 8       # 径向基函数数量
        self.num_cutoff_basis: int = 5       # 截断基函数数量
        self.max_ell: int = 3                # 球谐函数的最大角动量阶数
        self.correlation: int = 3            # ACE 关联阶数
        
        # -------------------- 网络结构参数 --------------------
        self.hidden_irreps: str = "128x0e + 64x1o"   # 隐藏层等变不可约表示
        self.num_channels: int = 128                  # 通道数
        self.num_blocks: int = 2                      # 消息传递块数
        
        # -------------------- 损失函数权重 --------------------
        self.energy_weight: float = 1.0               # 能量损失权重
        self.force_weight: float = 100.0              # 力损失权重
        self.stress_weight: float = 0.01              # 应力损失权重
        
        # -------------------- 优化器参数 --------------------
        self.learning_rate: float = 1e-3
        self.weight_decay: float = 1e-5
        self.batch_size: int = 32
        
        # -------------------- 训练参数 --------------------
        self.max_num_epochs: int = 100
        self.save_every: int = 10
        self.eval_every: int = 1
        self.output_dir: str = "./outputs/mace"
        
        # -------------------- 数据参数 --------------------
        self.train_path: str = "./data/xyz/train.xyz"
        self.val_path: str = "./data/xyz/valid.xyz"
        self.test_path: str = "./data/xyz/test.xyz"
        self.processed_dir: str = "./data/processed/"
        
        # 从字典更新配置
        if config_dict is not None:
            self.update(config_dict)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MACEConfig":
        """从字典创建配置实例"""
        return cls(config_dict)