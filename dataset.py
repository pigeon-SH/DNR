from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from util import DEVICE

def find_files(dir, exts=['*.png', '*.jpg']):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg'] or ['*.npy']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

def img_transform(image):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = image_transforms(image)
    return image

class SceneDataset(Dataset):
    def __init__(self, root, scene):
        self.ext_dir = os.path.join(root, scene, "extrinsics")
        self.uv_dir = os.path.join(root, scene, "uv")
        self.img_dir = os.path.join(root, scene, "frame")
        self.ext_files = find_files(self.ext_dir, exts=['*.npy'])        
        self.uv_files = find_files(self.uv_dir, exts=['*.npy'])
        self.img_files = find_files(self.img_dir, exts=['*.jpg', '*.png'])
        
    def __len__(self):
        return len(self.ext_files)

    def __getitem__(self, idx):
        #ext_path = os.path.join(self.ext_dir, self.ext_files[idx])
        #uv_path = os.path.join(self.uv_dir, self.uv_files[idx])
        #img_path = os.path.join(self.img_dir, self.img_files[idx])
        ext_path = self.ext_files[idx]
        uv_path = self.uv_files[idx]
        img_path = self.img_files[idx]
        
        ext = torch.FloatTensor(np.load(ext_path))
        uv = torch.FloatTensor(np.load(uv_path))
        img = img_transform(Image.open(img_path))
        
        ext = ext.to(device=DEVICE)
        uv = uv.to(device=DEVICE)
        img = img.to(device=DEVICE)
        
        return ext, uv, img