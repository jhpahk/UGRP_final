from enum import auto
import torch
from torch import optim
import torch.nn as nn

import time
import os

from encoder import AutoEncoder
from dataset import MPII
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


now = time.localtime()

time_path = 'checkpoint_%04d_%02d_%02d_%02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
checkpoint_dir_path = "./checkpoint/" + time_path
os.makedirs(checkpoint_dir_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
writer = SummaryWriter()


mpii_data = MPII()
train_loader = DataLoader(mpii_data, batch_size=32, shuffle=True)

autoencoder = AutoEncoder().cuda()
autoencoder = nn.DataParallel(autoencoder)

learning_rate = 1e-5
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

epochs = 100

for epoch in range(epochs):
    start = time.time()
    for data in train_loader:
        data = data.cuda()
        out = autoencoder(data)
        loss = loss_fn(data, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()

    time_cost = (end - start) / 60

    if (epoch + 1) % 10 == 0:
        checkpoint_file_path = time_path + f"_epoch_{epoch + 1}.pth"
        CHECKPOINT_PATH = os.path.join(checkpoint_dir_path, checkpoint_file_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': autoencoder.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, CHECKPOINT_PATH)

    writer.add_scalar("Train Loss", loss, epoch)

    print(f"Epoch: {epoch + 1}/{epochs} -> Loss: {loss} / Time: {time_cost} min")