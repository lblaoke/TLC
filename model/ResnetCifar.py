import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import random

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return F.normalize(x,dim=1).mm(F.normalize(self.weight,dim=0))

class BasicBlock(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = lambda x: x
        if stride!=1 or in_planes!=planes:
            self.planes = planes
            self.in_planes = in_planes
            self.shortcut = lambda x: F.pad(x[:,:,::2,::2],(0,0,0,0,(planes-in_planes)//2,(planes-in_planes)//2),"constant",0)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))+self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_s(nn.Module):
    def __init__(self,block,num_blocks,num_experts,num_classes,reweight_temperature=0.2):
        super(ResNet_s,self).__init__()

        self.in_planes = 16
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.eta = reweight_temperature

        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1s = nn.ModuleList([self._make_layer(block,16,num_blocks[0],stride=1) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes

        self.layer2s = nn.ModuleList([self._make_layer(block,32,num_blocks[1],stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block,64,num_blocks[2],stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes

        self.linears = nn.ModuleList([NormedLinear(64,num_classes) for _ in range(num_experts)])

        self.use_experts = list(range(num_experts))
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes
        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not module.weight.requires_grad:
                    module.eval()

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))

        outs = []
        self.logits = outs
        b0 = None
        self.w = [torch.ones(len(x),dtype=torch.bool,device=x.device)]

        for i in self.use_experts:
            xi = self.layer1s[i](x)
            xi = self.layer2s[i](xi)
            xi = self.layer3s[i](xi)
            xi = F.avg_pool2d(xi,xi.shape[3])
            xi = xi.flatten(1)
            xi = self.linears[i](xi)
            xi = xi*30
            outs.append(xi)

            # evidential
            alpha = torch.exp(xi)+1
            S = alpha.sum(dim=1,keepdim=True)
            b = (alpha-1)/S
            u = self.num_classes/S.squeeze(-1)

            # update w
            if b0 is None:
                C = 0
            else:
                bb = b0.view(-1,b0.shape[1],1)@b.view(-1,1,b.shape[1])
                C = bb.sum(dim=[1,2])-bb.diagonal(dim1=1,dim2=2).sum(dim=1)
            b0 = b
            self.w.append(self.w[-1]*u/(1-C))

        # dynamic reweighting
        exp_w = [torch.exp(wi/self.eta) for wi in self.w]
        exp_w = [wi/wi.sum() for wi in exp_w]
        exp_w = [wi.unsqueeze(-1) for wi in exp_w]

        reweighted_outs = [outs[i]*exp_w[i] for i in self.use_experts]
        return sum(reweighted_outs)
