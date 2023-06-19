import os

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset import ArcfaceDataset
from model import resnet_face18
from tool import get_tfms


class ArcfaceEvaluate:
    def __init__(
        self,
        root,
        out_root,
        epoch,
        weight_path,
        device,
        model=None,
        img_size=(128, 128),
        batch_size=16,
        norm_stats='imagenet',
    ):
        self.device = device
        self.root = root
        self.out_root = out_root
        self.epoch = epoch
        os.makedirs(self.out_root, exist_ok=True)

        logger.info('Evaluate:')
        if model is None:
            self.model = self.load_model(weight_path)
        else:
            self.model = model

        transform = get_tfms(img_size=img_size, phase='test', norm_stats=norm_stats)
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
        print(f"Loading Model checkpoint {weight_path} ...")
        model_state = torch.load(weight_path, map_location=self.device)
        
        model = resnet_face18(use_se=True)
        try:
            model.load_state_dict(model_state)
        except: 
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in model_state.items():
                name = k.replace('module.', '', 1)
                # name = k.replace('model.', '', 1)
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            
        model = model.to(self.device)
        model.eval()
        return model

    def get_loader(self, batch_size, transform):
        dataset = ArcfaceDataset(split='test', root=self.root, transforms=transform)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))

        np.random.seed(1235)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        return train_loader, val_loader
    
    def _preprocess(self, input):
        if isinstance(input, str):  # path
            input = cv2.imread(input, 0)
        elif isinstance(input, np.ndarray):
            input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        img_resize = cv2.resize(input, (self.img_size))
        img_lst = np.dstack((img_resize, np.fliplr(img_resize)))
        img_lst = img_lst.transpose((2, 0, 1))
        img_lst = img_lst[:, np.newaxis, :, :]
        image_nor = img_lst.astype(np.float32, copy=False)

        image_nor -= 127.5
        image_nor /= 127.5

        img_tensor = torch.from_numpy(image_nor)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def embedding_dataset(self):
        self.model.eval()
        torch.cuda.empty_cache()
        for idx, (input, target) in enumerate(self.train_loader):
            # input = self._preprocess(input)
            input = input.to(self.device)
            output = self.model(input)
            output = output.detach().cpu()

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
            output = output.detach().cpu()

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
        torch.save(info_embedding, self.out_root + f'/res_embedding_{self.epoch}.pt')

    def clasification(self):
        info_embedding = torch.load(self.out_root + f'/res_embedding_{self.epoch}.pt')

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
        
        print('Size of embedding:')
        print(pd.DataFrame({"size": size_dict.values()}, index=size_dict.keys()))
        print(("-" * 80))
        
        print('KNN classification:')
        self.KNN_classify.fit(embedding_train, target_train)
        KNN_pred = self.KNN_classify.score(embedding_val, target_val)
        print(f'Accuracy KNN: {KNN_pred}')
        print(("-" * 80))

        print('SVM classification:')
        self.SVM_classify.fit(embedding_train, target_train)
        SVM_pred = self.SVM_classify.score(embedding_val, target_val)
        print(f'Accuracy SVM: {SVM_pred}')
        print(("-" * 80))

        # print('Linear SVM classification:')
        # self.LinearSVM_classify.fit(embedding_train, target_train)
        # LSVM_pred = self.LinearSVM_classify.score(embedding_val, target_val)
        # print(f'Accuracy Linear SVM: {LSVM_pred}')
        # print(("-" * 80))

        print('Random forest classification:')
        self.RF_classify.fit(embedding_train, target_train)
        RF_pred = self.RF_classify.score(embedding_val, target_val)
        print(f'Accuracy Random Forest: {RF_pred}')


def main():
    pass


if __name__ == '__main__':
    main()
