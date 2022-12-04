import os
import torch
import torch.nn.functional as F

from dataset import UVDataset
from model import Model

import random

def main():
    torch.cuda.set_device(0)

    log_epoch = 50
    log_dir = "./logs/train"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = open("./logs/log_loss_train.txt", "w")

    dataset = UVDataset("./data/basketball")
    #dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=True)

    model = Model(1024,1024,16)
    step = 0

    lr=0.001
    eps=1e-8
    l2 = '0.01, 0.001, 0.0001, 0'.split(',')
    l2 = [float(x) for x in l2]
    betas = '0.9, 0.999'.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)
    optimizer = torch.optim.Adam([
        {'params': model.texture.layer1, 'weight_decay': l2[0], 'lr': lr},
        {'params': model.texture.layer2, 'weight_decay': l2[1], 'lr': lr},
        {'params': model.texture.layer3, 'weight_decay': l2[2], 'lr': lr},
        {'params': model.texture.layer4, 'weight_decay': l2[3], 'lr': lr},
        {'params': model.unet.parameters(), 'lr': 0.1 * lr}],
        betas=betas, eps=eps)
    model = model.to('cuda')
    model.train()
    torch.set_grad_enabled(True)
    criterion = torch.nn.L1Loss()

    print('Training started')
    for i in range(1, 50000):
        print('Epoch {}'.format(i))
        #adjust_learning_rate(optimizer, i, lr)
        for samples in dataloader:
            extrinsics, uv_maps, images = samples

            step += images.shape[0]
            optimizer.zero_grad()
            preds = model(uv_maps.cuda(), extrinsics.cuda())

            #loss1 = criterion(RGB_texture.cpu(), images)
            #loss2 = criterion(preds.cpu(), images)
            #loss = loss1 + loss2
            loss = criterion(preds.cpu(), images)
            loss.backward()
            optimizer.step()
            print('loss at step {}: {}'.format(step, loss.item()))
            log_file.write('loss at step {}: {}\n'.format(step, loss.item()))

        # save checkpoint
        if i % log_epoch == 0:
            print('Saving checkpoint step:{}'.format(step))
            torch.save(model, os.path.join(log_dir,'epoch_{}.pt'.format(step)))

if __name__ == '__main__':
    main()
