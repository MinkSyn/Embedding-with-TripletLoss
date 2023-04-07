import torch
import torchvision.transforms as T

from const import IMG_SIZE


def get_tfms(img_size=None, norm_stats=None):
    tfms = [T.ToTensor()]
    if img_size is None:
        img_size = IMG_SIZE
    if isinstance(img_size, int):
        tfms.append(T.Resize((img_size, img_size)))
    elif isinstance(img_size, (list, tuple)):
        tfms.append(T.Resize(img_size))
    if norm_stats is not None:
        tfms.append(T.Normalize(*norm_stats))
    return T.Compose(tfms)


def to_device(data, device):
    device = verify_device(device)
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def verify_device(device):
    if device != 'cpu' and torch.cuda.is_available():
        return device
    return 'cpu'
