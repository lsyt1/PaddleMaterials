import os
import paddle
from paddle.io import DataLoader
from typing import Optional
from .configuration import MACEConfig
from .modeling import MACEModel
from .dataset import MACEDataset
from .loss import MACELoss
from .utils import set_seed, save_checkpoint, load_checkpoint, compute_mae

class MACETrainer:
    def __init__(self, config: MACEConfig,
                 model: Optional[MACEModel] = None,
                 train_dataset: Optional[MACEDataset] = None,
                 val_dataset: Optional[MACEDataset] = None,
                 test_dataset: Optional[MACEDataset] = None,
                 output_dir: Optional[str] = None):
        self.config = config
        self.output_dir = output_dir or config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        paddle.set_device(paddle.get_device())
        set_seed(42)

        self.model = model or MACEModel(config)
        self.loss_fn = MACELoss(config)

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=config.learning_rate,
            parameters=self.model.parameters(),
            weight_decay=config.weight_decay
        )
        self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=config.learning_rate,
            T_max=config.max_num_epochs,
            eta_min=1e-6
        )
        self.optimizer.set_lr_scheduler(self.scheduler)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        if test_dataset is not None:
            self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            outputs = self.model(
                batch['atomic_numbers'], batch['positions'],
                batch['edge_index'], batch['edge_dist'], batch['edge_vec'],
                batch.get('batch', None)
            )
            loss_dict = self.loss_fn(outputs['energy'], outputs['forces'],
                                     batch['energy'], batch['forces'])
            loss = loss_dict['total_loss']
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with paddle.no_grad():
            for batch in self.val_loader:
                outputs = self.model(
                    batch['atomic_numbers'], batch['positions'],
                    batch['edge_index'], batch['edge_dist'], batch['edge_vec'],
                    batch.get('batch', None)
                )
                loss_dict = self.loss_fn(outputs['energy'], outputs['forces'],
                                         batch['energy'], batch['forces'])
                total_loss += loss_dict['total_loss'].item()
        return total_loss / len(self.val_loader)

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config.max_num_epochs
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={self.scheduler.get_lr():.6f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(self.model, self.optimizer, epoch, val_loss,
                                os.path.join(self.output_dir, 'best_model.pdparams'), self.scheduler)
            if (epoch+1) % self.config.save_every == 0:
                save_checkpoint(self.model, self.optimizer, epoch, train_loss,
                                os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pdparams'), self.scheduler)

    def evaluate(self, dataset_name='test'):
        loader = getattr(self, f'{dataset_name}_loader', None)
        if loader is None:
            raise ValueError(f'No loader for {dataset_name}')
        self.model.eval()
        energy_mae = 0.0
        force_mae = 0.0
        n = 0
        with paddle.no_grad():
            for batch in loader:
                outputs = self.model(
                    batch['atomic_numbers'], batch['positions'],
                    batch['edge_index'], batch['edge_dist'], batch['edge_vec'],
                    batch.get('batch', None)
                )
                energy_mae += compute_mae(outputs['energy'], batch['energy'])
                force_mae += compute_mae(outputs['forces'], batch['forces'])
                n += 1
        return {'energy_mae': energy_mae / n, 'force_mae': force_mae / n}
