import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from .imagenet_lt_data_loaders import LT_Dataset

class ImbalanceCIFAR100DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True):
        normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32,padding=4) ,
            transforms.RandomHorizontalFlip()   ,
            transforms.RandomRotation(15)       ,
            transforms.ToTensor()               ,
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])

        if training:
            self.dataset = IMBALANCECIFAR100(data_dir,train=True,download=True,transform=train_trsfm)
            self.val_dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)
            self.val_dataset = None

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 100

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
            batch_size  = 4096                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )

class ImbalanceCIFAR10DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True):
        normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()   ,
            transforms.RandomRotation(15)       ,
            transforms.ToTensor()               ,
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])

        if training:
            self.dataset = IMBALANCECIFAR10(data_dir,train=True,download=True,transform=train_trsfm)
            self.val_dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_trsfm)
            self.val_dataset = None

        # Uncomment to use OOD datasets
        self.OOD_dataset = None
        # self.OOD_dataset = datasets.SVHN(data_dir,split="test",download=True,transform=test_trsfm)
        # self.OOD_dataset = LT_Dataset('../ImageNet_LT/ImageNet_LT_open','../ImageNet_LT/ImageNet_LT_open.txt',train_trsfm)
        # self.OOD_dataset = LT_Dataset('../Places_LT/Places_LT_open','../Places_LT/Places_LT_open.txt',train_trsfm)

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 10

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
            batch_size  = 4096                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )