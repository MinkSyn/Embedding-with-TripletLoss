import argparse
import os
import time
from collections import OrderedDict

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import AddMarginProduct, ArcMarginProduct, SphereProduct
from config import Config
from const import STATS
from dataset import ArcfaceDataset
from model import resnet_face18
from torch.nn import DataParallel
from evaluate import ArcfaceEvaluate
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

        self.model = self.load_model(cfg['root']['ckpt'])

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

        metric_fc = self._get_arcface(
            cfg['arcface']['algo'], cfg['num_classes'], cfg['arcface']['params']
        )
        self.metric_fc = metric_fc.to(self.device)

        self.optimizer = self._get_optim(cfg['optimizer']['algo'])
        self.optim_hp = cfg['optimizer']['hparams']

        self.criterion = self._get_criterion(cfg['loss']['algo'])

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

    def load_model(self, weight_path):
        model = resnet_face18(use_se=False)
        model = DataParallel(model)
        model_state = torch.load(weight_path, map_location=self.device)

        try:
            model.load_state_dict(model_state)
        except:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.to(self.device)
        model.eval()
        return model

    def get_loaders(self):
        tfms = get_tfms(img_size=self.img_size, norm_stats=self.norm_stats)
        train_ds = ArcfaceDataset(
            split='test',
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
            torch.cuda.empty_cache()
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

                feature = self.model(input)
                output = self.metric_fc(feature, target)
                loss = self.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.detach().cpu()
                train_losses.append(loss)

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
            # logger.info(epoch_info)
            print(epoch_info)


            logger.info(f"Save checkpoints for epoch {epoch + 1}.")
            weight_path = (
                f"{self.weights_dir}/{self.run_name}__ep{str(epoch+1).zfill(2)}.pth"
            )

            # Checkpoints
            if (epoch + 1) % 2 == 0:
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
            if (epoch + 1) % 3== 0:
                with torch.no_grad():
                    eval = ArcfaceEvaluate(
                        root=self.test_dir,
                        out_root=self.embedding_root,
                        weight_path=None,
                        epoch=epoch,
                        model=self.model,
                        img_size=self.img_size,
                        device=self.device,
                        batch_size=self.batch_size,
                        norm_stats=self.norm_stats,
                    )
                    eval.embedding_dataset()
                    eval.clasification()
                    # logger.info("-" * 20)
            print(("-" * 80))

            # Early stopping
            if train_loss <= self.early_stopping and epoch > 3:
                logger.info(f"Early stopping at epoch [{epoch}]")
                break

        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_time))
        logger.info(f"Total training time: {train_time}")

    @staticmethod
    def _get_arcface(algo, num_classes, params):
        if algo == 'add':
            return AddMarginProduct(512, num_classes, s=params['s'], m=params['m'])
        elif algo == 'arc':
            return ArcMarginProduct(
                512,
                num_classes,
                s=params['s'],
                m=params['m'],
                easy_margin=params['easy_margin'],
            )
        elif algo == 'sphere':
            return SphereProduct(512, num_classes, m=params['m'])
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

    def _get_criterion(self, crit_name=None):
        if crit_name == 'focal':
            # https://github.com/AdeelH/pytorch-multi-class-focal-loss
            focal_loss = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                # alpha=torch.tensor([.25]),
                alpha=None,
                # device=self.device,
                gamma=2,
                reduction='mean',
                force_reload=False
            )
            return focal_loss
        else:
            return torch.nn.CrossEntropyLoss()


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
