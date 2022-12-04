import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

def img_transform(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = image_transforms(image)
    return image


def map_transform(map):
    map = torch.from_numpy(map)
    return map


class UVDataset(Dataset):

    def __init__(self, dir, view_direction=True):
        self.idx_list = ['{:04d}'.format(i) for i in range(899)]
        self.dir = dir
        self.view_direction = view_direction

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.dir, 'frame/'+self.idx_list[idx]+'.png'), 'r'))
        uv_map = np.load(os.path.join(self.dir, 'uv/'+self.idx_list[idx]+'.npy'))
        nan_pos = np.isnan(uv_map)
        uv_map[nan_pos] = 0
        if np.any(np.isnan(uv_map)):
            print('nan in dataset')
        if np.any(np.isinf(uv_map)):
            print('inf in dataset')
        extrinsics = np.load(os.path.join(self.dir, 'extrinsics/'+self.idx_list[idx]+'.npy'))
        img, uv_map = img_transform(img), map_transform(uv_map)
        return extrinsics, uv_map, img