import torch
import torchvision.transforms as T

from const import IMG_SIZE


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def is_image_file(filename: str):
    return filename.lower().endswith(IMG_EXTENSIONS)


def get_tfms(img_size=(128, 128), phase='train', norm_stats=None):
    normalize = T.Normalize(mean=[0.5], std=[0.5])

    if phase == 'train':
        transforms = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transforms = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            normalize
        ])
    return transforms


def to_device(data, device):
    device = verify_device(device)
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'
