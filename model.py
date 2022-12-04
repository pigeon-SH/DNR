import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from texture import NeuralTexture
from unet import UNet
from util import DEVICE


class Model(nn.Module):
    def __init__(self, texH, texW, n_feat, n_level):
        super(Model, self).__init__()
        self.texture = NeuralTexture(texH, texW, n_feat, n_level).to(device=DEVICE)
        self.unet = UNet(n_feat, 3).to(device=DEVICE)
    
    def _spherical_harmonics_basis(self, extrinsics):
        '''
        extrinsics: a tensor shaped (N, 3)
        output: a tensor shaped (N, 9)
        '''
        batch = extrinsics.shape[0]
        sh_bands = torch.ones((batch, 9), dtype=torch.float).to(device=DEVICE)
        coff_0 = 1 / (2.0*math.sqrt(np.pi))
        coff_1 = math.sqrt(3.0) * coff_0
        coff_2 = math.sqrt(15.0) * coff_0
        coff_3 = math.sqrt(1.25) * coff_0
        # l=0
        sh_bands[:, 0] = coff_0
        # l=1
        sh_bands[:, 1] = extrinsics[:, 1] * coff_1
        sh_bands[:, 2] = extrinsics[:, 2] * coff_1
        sh_bands[:, 3] = extrinsics[:, 0] * coff_1
        # l=2
        sh_bands[:, 4] = extrinsics[:, 0] * extrinsics[:, 1] * coff_2
        sh_bands[:, 5] = extrinsics[:, 1] * extrinsics[:, 2] * coff_2
        sh_bands[:, 6] = (3.0 * extrinsics[:, 2] * extrinsics[:, 2] - 1.0) * coff_3
        sh_bands[:, 7] = extrinsics[:, 2] * extrinsics[:, 0] * coff_2
        sh_bands[:, 8] = (extrinsics[:, 0] * extrinsics[:, 0] - extrinsics[:, 1] * extrinsics[:, 1]) * coff_2   # changed 2 to 1 from unoffical github
        return sh_bands
    
    def forward(self, uv, ext):
        """
        uv: (N, H, W, 2)
        ext: (N, 3)
        """
        sampled_texture = self.texture(uv)
        SH = self._spherical_harmonics_basis(ext).unsqueeze(-1).unsqueeze(-1)
        sampled_texture[:, 3:12, ...] = sampled_texture[:, 3:12, ...] * SH
        rgb = self.unet(sampled_texture)
        return rgb