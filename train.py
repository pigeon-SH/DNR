import os
import torch
import torch.nn.functional as F

from dataset import SceneDataset
from model import Model

def train(batch_size, n_epochs=50000, lr=0.001):
    
    log_dir = "./logs"
    save_dir = os.path.join(log_dir, "train_ckpts")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    log_iter = 10
    save_iter = 10000
    log_file = open(os.path.join(log_dir, "train_loss.txt"), "w")
    
    dataset = SceneDataset('./data', 'basketball', split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Model(1024, 1024, 16, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = F.l1_loss
    
    global_iter = n_epochs
    iters = 0
    
    print("TRAIN START")
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
        
            if iters >= global_iter:
                break
            if iters % log_iter == 0:
                print("step:{:6d}   loss:{:6.3f}".format(iters, loss.item()))
                log_file.write("step:{:6d}   loss:{:6.3f}\n".format(iters, loss.item()))
            if iters % save_iter == 0:
                torch.save(model, os.path.join(save_dir, "model_{}.pt".format(iters)))
    
    print("TRAIN DONE")

if __name__ == "__main__":
    train(batch_size=10)