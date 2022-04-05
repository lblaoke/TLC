import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import autocast

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,block,layers,num_experts,num_classes=1000,layer3_output_dim=None,layer4_output_dim=None,reweight_temperature=0.5):
        self.inplanes = 64
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.eta = reweight_temperature
        self.use_experts = list(range(num_experts))

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        if layer3_output_dim is None:
            layer3_output_dim = 256
        if layer4_output_dim is None:
            layer4_output_dim = 512

        self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.linears = nn.ModuleList([NormedLinear(layer4_output_dim * 4, num_classes) for _ in range(num_experts)])

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1
        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride!=1 or self.inplanes!=planes*4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes*4),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes*4
        for i in range(1,blocks):
            layers.append(block(self.next_inplanes, planes))
        return nn.Sequential(*layers)

    def expert_forward(self,x,ind):
        x = self.layer4s[ind](x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.linears[ind](x)
        return x*30

    def forward(self, x):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            outs = []
            self.logits = outs
            b0 = None
            self.w = [torch.ones(len(x),dtype=torch.bool,device=x.device)]

            for i in range(self.num_experts):
                xi = self.expert_forward(x,i)
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
