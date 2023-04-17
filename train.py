import argparse
import os
import time
from collections import OrderedDict

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from const import STATS
from loss import TripletLoss
from dataset import PatchCoreDataset
from model import ResNet50_v4
from evaluate import PatchCoreEvaluate
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

        model = ResNet50_v4(arch=self.model_arch, pretrained=True, testing=False)
        self.model = model.to(self.device)

        self.data_root = cfg['root']['data']
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

        self.criterion = TripletLoss(device=self.device)

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
        if cfg['root']['ckpt'] is not None:
            self.ckpt_path = cfg['root']['ckpt']

    def get_loaders(self):
        tfms = get_tfms(img_size=self.img_size, norm_stats=self.norm_stats)
        train_ds = PatchCoreDataset(
            root=self.data_root,
            transforms=tfms,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_dl

    def fit(self):
        torch.cuda.empty_cache()

        logger.info("Training...")
        train_time = time.time()

        train_dl = self.get_loaders()
        optimizer = self.optimizer(self.model.parameters(), **self.optim_hp)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.sched_hp)
        else:
            scheduler = None

        # Init
        init_epoch = 0

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
            epoch_time = time.time()

            # Training phase
            self.model.train()
            train_losses = []

            # for i, batch in enumerate(
            #     tqdm(train_dl, desc=f"Training epoch {(epoch + 1):>2d}")
            # ):
            logger.info(f"Training epoch {(epoch + 1):>2d}")
            for i, batch in enumerate(train_dl):
                input, target = batch
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)

                loss = self.criterion(output, target)
                loss = loss.detach().cpu()
                train_losses.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = torch.stack(train_losses).mean().item()

            if scheduler is not None:
                scheduler.step()
                last_lr = scheduler.optimizer.param_groups[0]['lr']
            else:
                last_lr = optimizer.param_groups[0]['lr']

            epoch_time = time.strftime(
                '%H:%M:%S', time.gmtime(time.time() - epoch_time)
            )
            epoch_info = (
                f"Epoch [{epoch:>2d}] : time: {epoch_time} | "
                f"last_lr: {last_lr:.6f} | train_loss: {train_loss:.4f}"
            )
            logger.info(epoch_info)

            # Checkpoints
            logger.info(f"Save checkpoints for epoch {epoch + 1}.")
            weight_path = (
                f"{self.weights_dir}/{self.run_name}__ep{str(epoch+1).zfill(2)}.pth"
            )
            embedding_root = (
                f"{self.out_root}/epoch_{epoch+1}"
            )
            os.makedirs(embedding_root, exist_ok=True)

            if (epoch + 1) % 1 == 0:
                print(f"Save weight epoch {epoch}")
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
                # Evaluate
                eval = PatchCoreEvaluate(
                    root=self.data_root,
                    out_root=embedding_root,
                    weight_path=weight_path,
                    img_size=self.img_size,
                    device=self.device,
                    batch_size=self.batch_size,
                    norm_stats=self.data_ver,
                )
                eval.embedding_dataset()
                eval.clasification()
                
            # Early stopping
            if train_loss <= self.early_stopping and epoch > 3:
                logger.info(f"Early stopping at epoch [{epoch}]")
                break

            if (epoch + 1) % 5 == 0:
                logger.info("-" * 20)

        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_time))
        logger.info(f"Total training time: {train_time}")

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
