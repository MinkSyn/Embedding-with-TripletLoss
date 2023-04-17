import os

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import PatchCoreDataset
from model import ResNet50_v4
from tool import get_tfms

DATA_ROOT = 'D:/datasets/Anodet_ICQC/Testing/Syn_v3.6'
EMBED_DIR = f'E:/PatchCore-Private/embedding'


class PatchCoreEvaluate:
    def __init__(
        self,
        root,
        out_root,
        weight_path,
        device,
        img_size=(300, 450),
        batch_size=16,
        norm_stats='patchcore_v3.6',
    ):
        self.device = device
        self.root = root
        self.out_root = out_root
        os.makedirs(self.out_root, exist_ok=True)

        model = self.load_model(weight_path)
        self.model = model.to(self.device)

        transform = get_tfms(img_size=img_size, norm_stats=norm_stats)
        self.train_loader, self.val_loader = self.get_loader(
            batch_size=batch_size, transform=transform
        )

        self.KNN_classify = KNeighborsClassifier()
        self.SVM_classify = SVC()
        self.LinearSVM_classify = LinearSVC()
        self.RF_classify = RandomForestClassifier()

    def load_model(self, weight_path):
        if not os.path.exists(weight_path):
            raise Exception(f'Not exits path: {weight_path}')
        print(f"Loading PatchCore checkpoint {weight_path} ...")
        ckpt = torch.load(weight_path)

        model = ResNet50_v4(arch=ckpt['arch'], testing=False)
        model = model.load_state_dict(ckpt['model_state'])
        return model

    def get_loader(self, batch_size, transform):
        dataset = PatchCoreDataset(self.root, transform)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.8 * dataset_size))

        np.random.seed(1235)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, val_loader

    def embedding_dataset(self):
        self.model.train()
        for idx, (input, target) in enumerate(self.train_loader):
            input = input.to(self.device)
            output = self.model(input)
            
            if idx == 0:
                embedding_train = output
                target_train = target
            else:
                embedding_train = torch.cat(
                    (embedding_train, output),
                    dim=0,
                )
                target_train = torch.cat(
                    (target_train, target),
                    dim=0,
                )

        for idx, (input, target) in enumerate(self.val_loader):
            input = input.to(self.device)
            output = self.model(input)

            if idx == 0:
                embedding_val = output
                target_val = target
            else:
                embedding_val = torch.cat(
                    (embedding_val, output),
                    dim=0,
                )
                target_val = torch.cat(
                    (target_val, target),
                    dim=0,
                )

        info_embedding = {
            'embedding_train': embedding_train.numpy(),
            'target_train': target_train.numpy(),
            'embedding_val': embedding_val.numpy(),
            'target_val': target_val.numpy(),
        }
        torch.save(info_embedding, self.out_root + 'info_embedding.pt')

    def clasification(self):
        info_embedding = torch.load(self.out_root + 'info_embedding.pt')

        embedding_train = info_embedding['embedding_train']
        target_train = info_embedding['target_train']

        embedding_val = info_embedding['embedding_val']
        target_val = info_embedding['target_val']

        size_dict = {
            'train datasets': embedding_train.shape,
            'train labels': target_train.shape,
            'valid datasets': embedding_val.shape,
            'valid labels': target_val.shape,
        }

        logger.info('Size of embedding:')
        print(pd.DataFrame({"size": size_dict.values()}, index=size_dict.keys()))

        logger.info('KNN classification:')
        self.KNN_classify.fit(embedding_train, target_train)
        KNN_pred = self.KNN_classify.score(embedding_val, target_val)
        print(f'Accuracy KNN: {KNN_pred}')

        logger.info('SVM classification:')
        self.SVM_classify.fit(embedding_train, target_train)
        SVM_pred = self.SVM_classify.score(embedding_val, target_val)
        print(f'Accuracy SVM: {SVM_pred}')

        logger.info('Linear SVM classification:')
        self.LinearSVM_classify.fit(embedding_train, target_train)
        LSVM_pred = self.LinearSVM_classify.score(embedding_val, target_val)
        print(f'Accuracy Linear SVM: {LSVM_pred}')

        logger.info('Random forest classification:')
        self.RF_classify.fit(embedding_train, target_train)
        RF_pred = self.RF_classify.score(embedding_val, target_val)
        print(f'Accuracy Random Forest: {RF_pred}')


def main():
    pass


if __name__ == '__main__':
    main()
