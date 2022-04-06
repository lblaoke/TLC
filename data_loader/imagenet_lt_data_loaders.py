import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image

class LT_Dataset(Dataset):
    def __init__(self,root,txt,transform):
        self.img_paths = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_paths.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        path,label = self.img_paths[index],self.labels[index]
        with open(path,'rb') as f:
            img = Image.open(f).convert('RGB')
            img = self.transform(img)
        return img,label

class ImageNetLTDataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True):
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224)                                       ,
            transforms.RandomHorizontalFlip()                                       ,
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0),
            transforms.ToTensor()                                                   ,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256)      ,
            transforms.CenterCrop(224)  ,
            transforms.ToTensor()       ,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # We use relative path to avoid potential bugs. It is recommended to check the paths below to ensure data loading.
        if training:
            self.dataset = LT_Dataset('../ImageNet_LT','../ImageNet_LT/ImageNet_LT_train.txt',train_trsfm)
            self.val_dataset = LT_Dataset('../ImageNet_LT','../ImageNet_LT/ImageNet_LT_val.txt',test_trsfm)
        else: # test
            self.dataset = LT_Dataset(data_dir,data_dir+'/ImageNet_LT_val.txt',test_trsfm)
            self.val_dataset = None

        self.n_samples = len(self.dataset)

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 1000

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : False
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=None)

    def split_validation(self,type='test'):
        return DataLoader(
            dataset     = self.OOD_dataset if type=='OOD' else self.val_dataset ,
            batch_size  = 512                                                   ,
            shuffle     = False                                                 ,
            num_workers = 10                                                    ,
            drop_last   = False
        )
