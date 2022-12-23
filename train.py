import os
import torch
import torch.nn.functional as F
import tqdm
from torchvision.transforms.functional import to_pil_image

import util
from dataset import SceneDataset
from model import Model


def train(batch_size, n_epochs=50000, lr=0.001):
    
    log_dir = "./logs"
    save_dir = os.path.join(log_dir, "train_ckpts")
    val_dir = os.path.join(log_dir, "validation")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
        
    log_iter = 10
    val_iter = 1000
    save_iter = 10000
    log_file = open(os.path.join(log_dir, "train_loss.txt"), "w")
    val_log_file = open(os.path.join(val_dir, "val_psnr.txt"), "w")
    
    dataset = SceneDataset('./data', 'basketball', split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_dataset = SceneDataset('./data', 'basketball', split="test")
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_data = val_dataset[0]
    gt_img = val_data[-1]
    gt_img = to_pil_image(gt_img, "RGB")
    gt_img.save(os.path.join(val_dir, "val_gt.png"))
    
    model = Model(1024, 1024, 16, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = F.l1_loss
    
    global_iter = n_epochs
    iters = 0
    
    print("TRAIN START")
    with tqdm.tqdm(total=global_iter) as pbar:
        while iters < global_iter + 1:
            for idx, data in enumerate(dataloader):
                iters += 1
                ext, uv, gt_img = data
                pred = model(uv, ext)
                #import pdb;pdb.set_trace()
                loss = loss_fn(pred, gt_img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if iters % log_iter == 0:
                    #print("step:{:6d}   loss:{:6.3f}".format(iters, loss.item()))
                    tqdm.tqdm.write("step:{:6d}   loss:{:6.3f}".format(iters, loss.item()))
                    log_file.write("step:{:6d}   loss:{:6.3f}\n".format(iters, loss.item()))
                    
                if iters % val_iter == 0:
                    with torch.no_grad():
                        ext, uv, gt_img = val_data
                        pred = model(uv, ext)
                        
                        psnr = util.PSNR(pred, gt_img)
                        print("step:{:6d}   psnr:{:6.3f}".format(idx, psnr.item()))
                        val_log_file.write("step:{:6d}   psnr:{:6.3f}\n".format(idx, psnr.item()))
                        
                        pred_img = to_pil_image(pred[0], "RGB")
                        pred_img.save(os.path.join(val_dir, "{}_pred.png".format(iters)))
                        
                        
                if iters % save_iter == 0:
                    torch.save(model, os.path.join(save_dir, "model_{}.pt".format(iters)))
                if iters >= global_iter:
                    break
                
                pbar.update(1)
    
    print("TRAIN DONE")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    train(batch_size=1, n_epochs=10000)