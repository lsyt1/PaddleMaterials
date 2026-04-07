import os
import sys
import argparse
import yaml
import paddle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paddle_materials.models.mace.configuration import MACEConfig
from paddle_materials.models.mace.modeling import MACEModel
from paddle_materials.models.mace.dataset import MACEDataset
from paddle_materials.models.mace.utils import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    config = MACEConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            if 'model' in cfg:
                for k,v in cfg['model'].items():
                    setattr(config, k, v)
    config.batch_size = args.batch_size
    dataset = MACEDataset(args.data, config, processed_path=None, train=False)
    model = MACEModel(config)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    loader = paddle.io.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    energies = []
    with paddle.no_grad():
        for batch in loader:
            out = model(batch['atomic_numbers'], batch['positions'],
                        batch['edge_index'], batch['edge_dist'], batch['edge_vec'],
                        batch.get('batch', None))
            energies.extend(out['energy'].numpy().tolist())
    print(f"Predicted {len(energies)} energies. Range: {min(energies):.4f} ~ {max(energies):.4f} eV")
    import csv
    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index','energy'])
        for i,e in enumerate(energies):
            writer.writerow([i,e])
    print(f"Saved to {args.output}")

if __name__ == '__main__':
    main()