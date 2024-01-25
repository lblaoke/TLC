import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp

class TLCLoss(nn.Module):
    def __init__(self,cls_num_list=None,max_m=0.5,reweight_epoch=-1,reweight_factor=0.05,annealing=500,tau=0.54):
        super(TLCLoss,self).__init__()
        self.reweight_epoch = reweight_epoch

        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = torch.tensor(m_list,dtype=torch.float,requires_grad=False)
        self.m_list = m_list

        if reweight_epoch!=-1:
            idx = 1
            betas = [0,0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights_enabled = None
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)

        # save diversity per_cls_weights
        self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights,dtype=torch.float,requires_grad=False).to("cuda:0")
        self.T = (reweight_epoch+annealing)/reweight_factor
        self.tau = tau

    def to(self,device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)
        return self

    def _hook_before_epoch(self,epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch
            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self,x,y):
        index = torch.zeros_like(x,dtype=torch.uint8,device=x.device)
        index.scatter_(1,y.data.view(-1,1),1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None,:],index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x-30*batch_m
        return torch.exp(torch.where(index,x_m,x))

    def forward(self,x,y,epoch,extra_info=None):
        loss = 0
        for i in range(extra_info["num_expert"]):
            alpha = self.get_final_output(extra_info["logits"][i],y)
            S = alpha.sum(dim=1,keepdim=True)
            l = F.nll_loss(torch.log(alpha)-torch.log(S),y,weight=self.per_cls_weights_base,reduction="none")

            # KL
            yi = F.one_hot(y,num_classes=alpha.shape[1])

            # adjusted parameters of D(p|alpha)
            alpha_tilde = yi+(1-yi)*(alpha+1)
            S_tilde = alpha_tilde.sum(dim=1,keepdim=True)
            kl = torch.lgamma(S_tilde)-torch.lgamma(torch.tensor(alpha_tilde.shape[1]))-torch.lgamma(alpha_tilde).sum(dim=1,keepdim=True) \
                +((alpha_tilde-1)*(torch.digamma(alpha_tilde)-torch.digamma(S_tilde))).sum(dim=1,keepdim=True)
            l += epoch/self.T*kl.squeeze(-1)

            # diversity
            if self.per_cls_weights_diversity is not None:
                diversity_temperature = self.per_cls_weights_diversity.view((1,-1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = 1
                temperature_mean = 1
            output_dist = F.log_softmax(extra_info["logits"][i]/diversity_temperature,dim=1)
            with torch.no_grad():
                mean_output_dist = F.softmax(x/diversity_temperature,dim=1)
            l -= 0.01*temperature_mean*temperature_mean*F.kl_div(output_dist,mean_output_dist,reduction="none").sum(dim=1)

            # dynamic engagement
            w = extra_info['w'][i]/extra_info['w'][i].max()
            w = torch.where(w>self.tau,True,False)
            l = (w*l).sum()/w.sum()
            loss += l.mean()

        return loss
