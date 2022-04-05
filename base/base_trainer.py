import torch
import torch.nn
from abc import abstractmethod
from numpy import inf
from utils import load_state_dict, rename_parallel_state_dict

class BaseTrainer:
    def __init__(self,model,criterion,opt,config):

        # Check with nvidia-smi about the available GPUs. Only 1 GPU is required.
        self.device = torch.device('cuda:0')

        self.config = config
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.opt = opt
        self.epochs = config['trainer']['epochs']

    @abstractmethod
    def _train_epoch(self,epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(1,self.epochs+1):
            result = self._train_epoch(epoch)
