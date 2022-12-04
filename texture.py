import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from util import DEVICE

"""
Difference with unofficial github
1. initialize texture with value range [0, 1] uniform distribution
    but unofficial github used torch.FloatTensor which generates random values range of entire float32
2. unofficial github generates each pyramid layers independently
    but I firstly generate texture of original resolution and blur&downsampling it to smaller size
"""

class LaplacianPyramid(nn.Module):
    def __init__(self, H, W, n_feat=16, K=4):
        """
        H, W, K: max_height, max_width, level
        n_feat: feature_dim
        """
        super(LaplacianPyramid, self).__init__()
        self.K = K
        #self.layers = [nn.Parameter(torch.rand((1, n_feat, H // (2**i), W // (2**i)))).to(device=DEVICE) for i in range(K)]

        self.texture = nn.Parameter(torch.rand((1, n_feat, H, W))).to(device=DEVICE)
        self.layers = [self.texture]
        layer_np = self.texture[0].permute((1, 2, 0)).cpu().detach().numpy()  # (H, W, n_feat)
        for i in range(1, K):
            layer_np = cv2.pyrDown(layer_np, dstsize=(H // (2**i), W // (2**i)))
            layer_tensor = torch.FloatTensor(layer_np).unsqueeze(0)  # (1, H, W, n_feat)
            layer_tensor = nn.Parameter(layer_tensor.permute((0, 3, 1, 2))).to(device=DEVICE)
            self.layers.append(layer_tensor)
    
    def forward(self, x):
        """
        x: uv (N, H, W, 2)
        output: sampled_texture (N, H, W, n_feat)
        """
        batch_size = len(x)
        #import pdb; pdb.set_trace()
        x = [F.grid_sample(torch.repeat_interleave(self.layers[i], batch_size, 0), x, align_corners=False) for i in range(self.K)]
        x = sum(x)
        return x

class NeuralTexture(nn.Module):
    def __init__(self, H=512, W=512, n_feat=16, K=4):
        super(NeuralTexture, self).__init__()
        self._texture = LaplacianPyramid(H, W, n_feat, K)
    
    def forward(self, x):
        """
        x: uv (N, H, W, 2)
        """
        return self._texture(x)

    def cuda(self):
        for i in range(len(self._texture.layers)):
            self._texture.layers[i].cuda()
    
    def cpu(self):
        for i in range(len(self._texture.layers)):
            self._texture.layers[i].cpu()