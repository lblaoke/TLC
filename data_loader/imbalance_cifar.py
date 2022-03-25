import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from math import *

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    num_class = 10
    decay_stride = 2.1971

    def __init__(self,root,imb_type='exp',train=True,transform=None,target_transform=None,download=False):
        super(IMBALANCECIFAR10,self).__init__(root, train, transform, target_transform, download)
        img_num_list = self.get_img_num_per_cls(self.num_class,imb_type)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self,num_class,imb_type):
        img_max = len(self.data)/num_class
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(num_class):
                num = img_max*exp(-cls_idx/self.decay_stride)
                img_num_per_cls.append(int(num+0.5))
        else:
            img_num_per_cls.extend([int(img_max)]*num_class)
        return img_num_per_cls

    def gen_imbalanced_data(self,img_num_per_cls):
        img_max = len(self.data)/self.num_class
        new_data,new_targets = [],[]
        targets_np = np.array(self.targets,dtype=np.int64)
        classes = np.arange(self.num_class)

        self.num_per_cls = np.zeros(self.num_class)
        for class_i,volume_i in zip(classes,img_num_per_cls):
            self.num_per_cls[class_i] = volume_i
            idx = np.where(targets_np==class_i)[0]
            np.random.shuffle(idx)
            keep_num = volume_i+1
            selec_idx = idx[:keep_num]
            new_data.append(self.data[selec_idx,...])
            new_targets.extend([class_i]*keep_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        return self.num_per_cls.tolist()

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]
    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_class = 100
    decay_stride = 21.9714
