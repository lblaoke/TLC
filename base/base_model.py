import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
