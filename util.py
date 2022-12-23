import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIXEL_MAX = 255.0

def PSNR(pred, gt_img):
    mse = torch.mean((pred - gt_img) ** 2)
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return psnr