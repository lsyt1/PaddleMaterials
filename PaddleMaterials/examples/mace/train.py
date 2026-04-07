import os
import sys
import argparse
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paddle_materials.models.mace.configuration import MACEConfig
from paddle_materials.models.mace.dataset import MACEDataset
from paddle_materials.models.mace.trainer import MACETrainer
from paddle_materials.models.mace.utils import load_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--do_eval', action='store_true')
    p.add_argument('--do_test', action='store_true')
    p.add_argument('--checkpoint', default=None)
    return p.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    config = MACEConfig(cfg_dict.get('model', {}))
    if 'optimizer' in cfg_dict:
        for k,v in cfg_dict['optimizer'].items():
            setattr(config, k, v)
    if 'training' in cfg_dict:
        for k,v in cfg_dict['training'].items():
            setattr(config, k, v)

    # 修正路径：基于 examples/mace 目录，而不是 configs 目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))  # 上一级目录即 examples/mace
    def get_path(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    data_cfg = cfg_dict.get('data', {})
    train_ds = None
    val_ds = None
    test_ds = None

    if not args.do_eval and not args.do_test:
        if 'train_path' in data_cfg:
            train_ds = MACEDataset(get_path(data_cfg['train_path']), config,
                                   processed_path=get_path(data_cfg.get('processed_dir','./data/processed/train.pt')))
        if 'val_path' in data_cfg:
            val_ds = MACEDataset(get_path(data_cfg['val_path']), config,
                                 processed_path=get_path(data_cfg.get('processed_dir','./data/processed/val.pt')))
    if args.do_eval and 'val_path' in data_cfg:
        val_ds = MACEDataset(get_path(data_cfg['val_path']), config,
                             processed_path=get_path(data_cfg.get('processed_dir','./data/processed/val.pt')))
    if args.do_test and 'test_path' in data_cfg:
        test_ds = MACEDataset(get_path(data_cfg['test_path']), config,
                              processed_path=get_path(data_cfg.get('processed_dir','./data/processed/test.pt')))

    trainer = MACETrainer(config, train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
                          output_dir=cfg_dict.get('training',{}).get('output_dir','./outputs/mace'))
    if args.checkpoint and os.path.exists(args.checkpoint):
        load_checkpoint(trainer.model, args.checkpoint, trainer.optimizer, trainer.scheduler)

    if args.do_test:
        metrics = trainer.evaluate('test')
        print(f"Test: Energy MAE = {metrics['energy_mae']:.6f} eV, Force MAE = {metrics['force_mae']:.6f} eV/Å")
    elif args.do_eval:
        metrics = trainer.evaluate('val')
        print(f"Val: Energy MAE = {metrics['energy_mae']:.6f} eV, Force MAE = {metrics['force_mae']:.6f} eV/Å")
    else:
        trainer.train()

if __name__ == '__main__':
    main()
