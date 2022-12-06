import os
import torch
import torch.nn.functional as F

from dataset import SceneDataset
from model import Model

def test():
    log_dir = "./logs"
    save_dir = os.path.join(log_dir, "test_result")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    dataset = SceneDataset('./data', 'basketball')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    model_dir = "./logs/train_ckpts"
    model_path = os.path.join(model_dir, sorted(os.listdir(model_dir))[-1])
    model = torch.load(model_path)
    
    for idx, data in enumerate(dataloader):
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

if __name__ == "__main__":
    test()