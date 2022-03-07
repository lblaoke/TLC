import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image

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

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

class ImageNetLTDataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers=0,training=True,retain_epoch_size=True):
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
            self.dataset = LT_Dataset('../ImageNet_LT','../ImageNet_LT/ImageNet_LT_train.txt', train_trsfm)
            self.val_dataset = LT_Dataset('../ImageNet_LT','../ImageNet_LT/ImageNet_LT_val.txt', test_trsfm)
        else: # test
            self.dataset = LT_Dataset(data_dir, data_dir + '/ImageNet_LT_val.txt', test_trsfm)
            self.val_dataset = None

        # Uncomment to use OOD datasets
        self.OOD_dataset = None
        # self.OOD_dataset = LT_Dataset('../ImageNet_LT/ImageNet_LT_open','../ImageNet_LT/ImageNet_LT_open.txt',train_trsfm)

        self.n_samples = len(self.dataset)

        num_classes = dataset.targets.max().item()+1
        assert num_classes == 1000

        self.cls_num_list = np.histogram(dataset.targets,bins=num_classes)[0].tolist()

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
            num_workers = 4                                                     ,
            drop_last   = False
        )
