import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker,load_state_dict,rename_parallel_state_dict,autocast
import model.model as module_arch
from model.metric import *
from tqdm import tqdm

class Trainer(BaseTrainer):
    def __init__(self,model,criterion,opt,config,data_loader,valid_data_loader,lr_scheduler):
        super().__init__(model,criterion,opt,config)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.val_targets = torch.tensor(valid_data_loader.dataset.targets,device=self.device).long()
        self.num_class = self.val_targets.max().item()+1
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self,epoch):
        self.model.train()
        self.model._hook_before_iter()
        self.criterion._hook_before_epoch(epoch)

        total_loss = []
        for batch_id,(data,target) in tqdm(enumerate(self.data_loader)):
            data,target = data.to(self.device),target.to(self.device)
            self.opt.zero_grad()

            with autocast():
                output = self.model(data)
                extra_info = {
                    "num_expert"    : len(self.model.backbone.logits)   ,
                    "logits"        : self.model.backbone.logits        ,
                    'w'             : self.model.backbone.w
                }
                loss = self.criterion(x=output,y=target,epoch=epoch,extra_info=extra_info)
                loss.backward()

            self.opt.step()
            total_loss.append(loss.item())

        self._valid_epoch(epoch)
        print("loss =",sum(total_loss)/len(total_loss))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _valid_epoch(self,epoch):
        self.model.eval()
        output = torch.empty(0,self.num_class,dtype=torch.float32,device=self.device)
        uncertainty = torch.empty(0,dtype=torch.float32,device=self.device)
        for _,(data,_) in enumerate(self.valid_data_loader):
            data = data.to(self.device)

            with torch.no_grad():
                o = self.model(data)
                u = self.model.backbone.w[-1]
            output = torch.cat([output,o.detach()],dim=0)
            uncertainty = torch.cat([uncertainty,u.detach()],dim=0)

        print(f'================ Epoch: {epoch:03d} ================')
        ACC(output,self.val_targets,uncertainty,region_len=self.num_class/3)
