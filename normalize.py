import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PatchCoreDataset
from tool import get_tfms

ROOT = 'D:/datasets/ICQC/patchcore_v3.6'
DATA_VER = 'patchcore_v3.6'


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(dataloader):
        # Mean over batch, height and width, but not over the channels (dim1)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean.tolist(), std.tolist()


def main():
    tfms = get_tfms(norm_stats=None)
    train_ds = PatchCoreDataset(
            root=ROOT,
            transforms=tfms,
        )
    train_dl = DataLoader(train_ds, batch_size=16, num_workers=4)
    
    logger.info(f"Calculating STATS for dataset [{DATA_VER}]...")
    mean, std = get_mean_and_std(train_dl)
    print(f"Mean: {[round(m, 4) for m in mean]}")
    print(f"Std: {[round(s, 4) for s in std]}")
    
if __name__ == '__main__':
    main()
    