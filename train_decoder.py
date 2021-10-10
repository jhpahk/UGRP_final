import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PoseTrackDataset
from encoder import AutoEncoder
from model_decoder import KeyPointDecoder

import time
import os


DATA_ROOT = "./data/"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # Imagenet mean, std
])

now = time.localtime()

time_path = 'check_point_%04d_%02d_%02d_%02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
checkpoint_dir_path = "./checkpoint_decoder/" + time_path
os.makedirs(checkpoint_dir_path)
checkpoint_file_path = time_path + ".pth"
CHECKPOINT_PATH = os.path.join(checkpoint_dir_path, checkpoint_file_path) 

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
writer = SummaryWriter()

train_loader = DataLoader(
                        PoseTrackDataset(
                            DATA_ROOT, 
                            transform,
                            limit=30
                        ),
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)


encoder = AutoEncoder()
encoder.load_state_dict(torch.load("autoencoder/UGRP_AutoEncoder/checkpoint/checkpoint_2021_10_09_04:42:23/checkpoint_2021_10_09_04:42:23_epoch_100.pth")['model_state_dict'])
encoder = encoder.encoder.cuda()

decoder = KeyPointDecoder()
decoder.cuda()
decoder = nn.DataParallel(decoder)

learning_rate = 1e-5
optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
end_epoch = 50
criterion = nn.MSELoss()

decoder.train()

for epoch in range(end_epoch):
    start = time.time()

    for i, data in enumerate(train_loader):
        loss = 0
        optimizer.zero_grad()

        (prev, kp_prev), (query, kp_query) = data
        prev = prev.cuda()
        query = query.cuda()

        prev_keymap = encoder(prev)
        query_keymap = encoder(query)

        query_heatmap = decoder(prev_keymap, kp_prev, query_keymap)
        if query_heatmap == None:
            continue

        query_heatmap = query_heatmap.reshape(query_heatmap.shape[0], 4096)
        query_heatmap = F.softmax(query_heatmap, dim=1)
        query_heatmap = query_heatmap.reshape(query_heatmap.shape[0], 64, 64)

        filtered_kp_query = []
        for k in range(len(kp_query)):
            if kp_prev[k] != 0:
                filtered_kp_query.append(kp_query[k])

        for k in range(len(filtered_kp_query)):
            if filtered_kp_query[k] == 0:
                continue

            x = filtered_kp_query[k][0][0] // 4
            y = filtered_kp_query[k][1][0] // 4
            loss += (1 - query_heatmap[k][y][x])**2


        if loss == 0:
            continue

        loss.backward()
        optimizer.step()

    end = time.time()
    time_cost = (end - start) / 60

    print(f"Epoch: {epoch + 1}/{end_epoch} -> Loss: {loss}, Time: {time_cost} min")

    if (epoch + 1) % 10 == 0:
        checkpoint_file_path = time_path + f"_epoch_{epoch + 1}.pth"
        CHECKPOINT_PATH = os.path.join(checkpoint_dir_path, checkpoint_file_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': decoder.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, CHECKPOINT_PATH)

    writer.add_scalar("Train Loss", loss, epoch)
