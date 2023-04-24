import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from const import CardID
from tool import is_image_file


class PatchCoreDataset(Dataset):
    def __init__(
        self,
        root: str,
        transforms: transforms = None,
    ):
        super().__init__()
        self.transforms = transforms

        self.samples = self.get_samples(root)

    # def get_samples(self, root):
    #     samples = []
    #     for card_type in os.listdir(root):
    #         id_class = CardID[card_type].value
    #         card_path = os.path.join(root, card_type)

    #         for quality_type in os.listdir(card_path):
    #             quality_path = os.path.join(card_path, quality_type)

    #             for file in os.listdir(quality_path):
    #                 filename = os.fsdecode(file)
    #                 if is_image_file(filename):
    #                     path = os.path.join(quality_path, filename)
    #                     samples.append((path, id_class))
    #     return samples
    
    def get_samples(self, root):
        samples = []
        for card_type in os.listdir(root):
            id_class = CardID[card_type].value
            card_path = os.path.join(root, card_type)

            for file in os.listdir(card_path):
                filename = os.fsdecode(file)
                if is_image_file(filename):
                    path = os.path.join(card_path, filename)
                    samples.append((path, id_class))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
