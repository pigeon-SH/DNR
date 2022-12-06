import os
import torch
import torch.nn.functional as F
import numpy as np

from dataset import SceneDataset
from model import Model

PIXEL_MAX = 255.0

def PSNR(pred, gt_img):
    mse = torch.mean((pred - gt_img) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr

def test():
    log_dir = "./logs"
    save_dir = os.path.join(log_dir, "test_result")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    log_file = open(os.path.join(save_dir, "test_log.txt"), "w")
    
    dataset = SceneDataset('./data', 'basketball', split="test")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    model_dir = "./logs/train_ckpts"
    model_path = os.path.join(model_dir, sorted(os.listdir(model_dir))[-1])
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            ext, uv, gt_img = data
            pred = model(uv, ext)
        
            psnr = PSNR(pred, gt_img)
            print("step:{:6d}   psnr:{:6.3f}".format(idx, psnr.item()))
            log_file.write("step:{:6d}   psnr:{:6.3f}\n".format(idx, psnr.item()))

if __name__ == "__main__":
    test()