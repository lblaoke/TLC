import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from . import ResnetImagenet
from . import ResnetCifar

class Model(BaseModel):
    requires_target = False

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self,x):
        return self.backbone(x)

class ResNet32Model(Model):
    def __init__(self,num_classes,num_experts=1,**kwargs):
        super().__init__(num_classes,None)
        self.backbone = ResnetCifar.ResNet_s(
            ResnetCifar.BasicBlock      ,
            [5,5,5]                     ,
            num_classes = num_classes   ,
            num_experts = num_experts   ,
            **kwargs
        )

class ResNet50Model(Model):
    def __init__(self,num_classes,layer3_output_dim=None,layer4_output_dim=None,num_experts=1,**kwargs):
        super().__init__(num_classes,None)
        self.backbone = ResnetImagenet.ResNet(
            ResnetImagenet.Bottleneck               ,
            [3,4,6,3]                               ,
            num_classes         = num_classes       ,
            layer3_output_dim   = layer3_output_dim ,
            layer4_output_dim   = layer4_output_dim ,
            num_experts         = num_experts       ,
            **kwargs
        )
