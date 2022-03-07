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

class CIFAR100DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,shuffle=True,num_workers=0,training=True):
        normalize = transforms.Normalize(mean=[0.5071,0.4865,0.4409],std=[0.2673,0.2564,0.2762])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])

        if training:
            self.dataset = datasets.CIFAR100(data_dir,train=True,download=True,transform=train_trsfm)
            self.val_dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)
        else:
            self.dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)

        num_classes = self.dataset.targets.max().item()+1
        assert num_classes == 100

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()
        self.n_samples = len(self.dataset)

        self.init_kwargs = {
            'batch_size'    : batch_size,
            'shuffle'       : shuffle   ,
            'num_workers'   : num_workers
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs)

    def split_validation(self):
        return DataLoader(dataset=self.val_dataset,**self.init_kwargs)

class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)
        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0]*self.bucket_num
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        for _ in range(self.__len__()):
            yield self._next_item()

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])
        else:
            return max([len(bucket) for bucket in self.buckets])*self.bucket_num

class ImbalanceCIFAR100DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,drop_last,training=True,retain_epoch_size=True):
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

        # Uncomment to use OOD datasets
        self.OOD_dataset = None
        # self.OOD_dataset = datasets.SVHN(data_dir,split="test",download=True,transform=test_trsfm)
        # self.OOD_dataset = LT_Dataset('../ImageNet_LT/ImageNet_LT_open','../ImageNet_LT/ImageNet_LT_open.txt',train_trsfm)
        # self.OOD_dataset = LT_Dataset('../Places_LT/Places_LT_open','../Places_LT/Places_LT_open.txt',train_trsfm)

        num_classes = self.dataset.targets.max().item()+1
        assert num_classes == 100

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : drop_last
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=sampler)

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

        num_classes = self.dataset.targets.max().item()+1
        assert num_classes == 10

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size,
            'shuffle'       : True      ,
            'num_workers'   : num_workers
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=sampler)

    def split_validation(self,type='test'):
        return DataLoader(
            dataset     = self.OOD_dataset if type=='OOD' else self.val_dataset ,
            batch_size  = 4096                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
