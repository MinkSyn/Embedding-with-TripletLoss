import argparse
import os
import time
from collections import OrderedDict

import torch
import timm
from loguru import logger
from sklearn.metrics import recall_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from const import STATS
from dataset import PatchCoreDataset
from tool import get_tfms, verify_device

torch.manual_seed(1235)


class Trainer:
    def __init__(self, cfg, resume=False):
        self.cfg = cfg
        self.resume = resume

        self.device = verify_device(cfg['device'])

        self.model_arch = cfg['model']['arch']
        logger.info(f"Constructing model {self.model_arch} ...")
        self.data_ver = cfg['data_ver']
        self.run_name = f"{self.model_arch}__{self.data_ver}"

        model = timm.create_model(
            arch=self.model_arch, pretrained=cfg['model']['pretrained']
        )
        self.model = model.to(self.device)

        self.data_root = cfg['root']['data']
        self.test_dir = cfg['root']['test']
        img_size = cfg['model']['img_size']
        if isinstance(img_size, int):
            self.img_size = img_size
        else:
            self.img_size = img_size['height'], img_size['width']

        self.norm_stats = STATS[self.data_ver]
        logger.info(f"Using stats of [{self.data_ver}]: {self.norm_stats}")

        self.batch_size = cfg['dataloader']['batch_size']
        self.num_workers = cfg['dataloader']['num_workers']
        self.pin_memory = cfg['dataloader']['pin_memory']

        self.num_epochs = cfg['hparams']['num_epochs']
        self.early_stopping = cfg['hparams']['early_stopping']

        self.optimizer = self._get_optim(cfg['optimizer']['algo'])
        self.optim_hp = cfg['optimizer']['hparams']

        self.criterion = self.get_loss_func(cfg['loss']['algo'])

        if 'scheduler' in cfg.keys():
            self.sched_algo = cfg['scheduler']['algo']
            self.scheduler = self._get_sched(self.sched_algo)
        else:
            self.scheduler = None
        self.sched_hp = cfg['scheduler']['hparams']

        self.out_root = f"{cfg['root']['out']}/run{cfg['run']}"
        os.makedirs(self.out_root, exist_ok=True)

        self.weights_dir = f"{self.out_root}/weights"
        os.makedirs(self.weights_dir, exist_ok=True)

        self.embedding_root = f"{self.out_root}/res_embedding"
        os.makedirs(self.embedding_root, exist_ok=True)
        if cfg['root']['ckpt'] is not None:
            self.ckpt_path = cfg['root']['ckpt']

    def get_loaders(self):
        tfms = get_tfms(img_size=self.img_size, norm_stats=self.norm_stats)
        train_ds = PatchCoreDataset(
            split='instances',
            root=f"{self.data_root}/train",
            transforms=tfms,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_ds = PatchCoreDataset(
            split='instances',
            root=f"{self.data_root}/val",
            transforms=tfms,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        test_ds = PatchCoreDataset(
            split='instances',
            root=self.test_dir,
            transforms=tfms,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_dl, val_dl, test_dl

    def fit(self):
        torch.cuda.empty_cache()

        logger.info("Training...")
        train_time = time.time()

        train_dl, valid_dl, test_dl = self.get_loaders()
        optimizer = self.optimizer(self.model.parameters(), **self.optim_hp)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.sched_hp)
        else:
            scheduler = None

        # Init
        init_epoch = 0
        patience = 10
        last_loss = 100
        trigger_times = 0

        if self.resume:
            ckpt = torch.load(self.ckpt_path)
            logger.info(f"Loading checkpoint from epoch [{ckpt['epoch'] + 1}]")
            init_epoch = ckpt['epoch'] + 1

            model_state = ckpt['model_state']
            new_model_state = OrderedDict()
            for k, v in model_state.items():
                name = k.replace('module.', '', 1)  # remove 'module.' of dataparallel
                new_model_state[name] = v
            self.model.load_state_dict(new_model_state)

            optimizer.load_state_dict(ckpt['optim_state'])
            if scheduler is not None:
                scheduler.load_state_dict(ckpt['sched_state'])

            logger.info(f"Resuming at epoch [{init_epoch}]")

        for epoch in range(init_epoch, self.num_epochs):
            torch.cuda.empty_cache()
            epoch_time = time.time()

            # Training phase
            self.model.train()
            train_losses = []
            for i, batch in enumerate(
                tqdm(train_dl, desc=f"Training epoch {(epoch + 1):>2d}")
            ):
                input, target = batch
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)
                batch_loss = self.criterion(output, target)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                batch_loss = batch_loss.detach().cpu()
                train_losses.append(batch_loss)
                
            self.model.eval()
            val_losses = []
            val_accs = []
            with torch.no_grad():
                for batch in tqdm(valid_dl, desc=f"Validating epoch {epoch:>2d}"):
                    input, target = batch
                    input = input.to(self.device)
                    target = target.to(self.device)

                    output = self.model(input)
                    batch_loss = self.criterion(output, target)
                    
                    _, preds = torch.max(output, dim=1)
                    acc = torch.tensor(torch.sum(preds == target).item() / len(preds))
                    batch_loss = batch_loss.detach().cpu()
                    
                    val_losses.append(batch_loss)
                    val_accs.append(acc)

            if scheduler is not None:
                scheduler.step()
                last_lr = scheduler.optimizer.param_groups[0]['lr']
            else:
                last_lr = optimizer.param_groups[0]['lr']

            epoch_time = time.strftime(
                '%H:%M:%S', time.gmtime(time.time() - epoch_time)
            )
            train_loss = torch.stack(train_losses).mean().item()
            val_loss = torch.stack(val_losses).mean().item()
            val_acc = torch.stack(val_accs).mean().item()
            
            epoch_info = (
                f"Epoch [{epoch:>2d}] : time: {epoch_time} | "
                f"last_lr: {last_lr:.6f} | train_loss: {train_loss:.4f} |"
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}"
            )
            # logger.info(epoch_info)
            print(epoch_info)

            logger.info(f"Save checkpoints for epoch {epoch + 1}.")
            weight_path = (
                f"{self.weights_dir}/{self.run_name}__ep{str(epoch+1).zfill(2)}.pth"
            )
            
            # Save best weight
            if not self.resume and epoch == 0:
                best_acc = val_acc
                best_loss = val_loss
            if val_acc >= best_acc and val_loss <= best_loss:
                logger.info(f"Found better weights at epoch [{epoch}].")
                best_acc = val_acc

                weight_path = f"{self.weights_dir}/{self.run_name}_ep{str(epoch).zfill(2)}.pth"
                torch.save(self.model.state_dict(), weight_path)

            # Checkpoints
            torch.save(
                {
                    'epoch': epoch,
                    'arch': self.model_arch,
                    'model_state': self.model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'sched_state': scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    'last_loss': train_loss,
                },
                weight_path,
            )
            
            # Early stopping
            current_loss = val_loss
            if current_loss > last_loss and epoch > 3:
                trigger_times += 1

                if trigger_times >= patience:
                    logger.info(f"Early stopping at epoch [{epoch}]")
                    break
            else:
                trigger_times = 0
                last_loss = current_loss
                
            # Evaluate
            if (epoch + 1) % 3 == 0:
                print(("-" * 80))
                test_accs = []
                targets, test_preds, normal_mcc = [], [], []

                with torch.no_grad():
                    for batch in tqdm(test_dl, desc=f"Testing epoch {epoch:>2d}"):
                        input, target = batch
                        input = input.to(self.device)
                        
                        output = self.model(input)
                        
                        _, batch_preds = torch.max(output, dim=1)
                        acc = torch.tensor(torch.sum(batch_preds == target).item() / len(batch_preds))
                        test_accs.append(acc)
                        
                        batch_preds = batch_preds.tolist()
                        targets += target.tolist()
                        test_preds += batch_preds
                        batch_preds[batch_preds == 0] = -1
                        normal_mcc += batch_preds
                    
                    test_acc = torch.stack(test_accs).mean().item()
                    test_recall = recall_score(targets, test_preds)
                    test_mcc = matthews_corrcoef(targets, normal_mcc)
                
                    logger.info(f"Testing accuracy: {test_acc}")
                    logger.info(f"Testing mcc: {test_recall}")
                    logger.info(f"Testing recall: {test_mcc}")
                    
            print(("-" * 80))

        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_time))
        logger.info(f"Total training time: {train_time}")

    def get_loss_func(self, loss_name):
        if loss_name == 'cross':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'focal':
            focal_loss = torch.hub.load(
                "adeelh/pytorch-multi-class-focal-loss",
                model="FocalLoss",
                # alpha=torch.tensor([.25]),
                alpha=None,
                # device=self.device,
                gamma=2,
                reduction="mean",
                force_reload=False,
            )
            return focal_loss
        else:
            raise NotImplementedError

    @staticmethod
    def _get_optim(optim_name):
        if optim_name == 'adam':
            return torch.optim.Adam
        elif optim_name == 'sgd':
            return torch.optim.SGD
        else:
            raise NotImplementedError

    @staticmethod
    def _get_sched(sched_name):
        if sched_name == 'one_cycle':
            return torch.optim.lr_scheduler.OneCycleLR
        elif sched_name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau
        elif sched_name == 'step':
            return torch.optim.lr_scheduler.StepLR
        else:
            raise NotImplementedError


def main(resume=False, config_path='config.yml'):
    config = Config.load_yaml(config_path)

    trainer = Trainer(config, resume)

    print(f"Current run: [{trainer.run_name}] on device [{trainer.device}]")
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', '-r', default=False, action='store_true')
    parser.add_argument('--config', '-c', default='config.yml')
    args = parser.parse_args()

    main(resume=args.resume, config_path=args.config)
