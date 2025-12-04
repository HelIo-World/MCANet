import torch.nn as nn
from torch import Tensor

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps = 0.00001, elementwise_affine = True, bias = True, device=None, dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
    
    def forward(self,input: Tensor) -> Tensor:
        input = input.permute(0,2,3,1)
        norm = super().forward(input)
        norm = input.permute(0,3,1,2)
        return norm