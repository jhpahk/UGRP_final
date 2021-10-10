import torch
from torch import optim
from torch.utils.data import DataLoader
from torch._C import _set_backcompat_keepdim_warn
import torchvision.transforms as transforms

# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from torchvision.io import read_image

from PIL import Image

import json

from model_autoencoder import EncoderEstimator
from dataset import PoseTrackDataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import time
import os


DATA_ROOT = "./data/"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # Imagenet mean, std
])

now = time.localtime()

time_path = 'checkpoint_%04d_%02d_%02d_%02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
checkpoint_dir_path = "./checkpoint/" + time_path
os.makedirs(checkpoint_dir_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
writer = SummaryWriter()


train_loader = DataLoader(
                        PoseTrackDataset(
                            DATA_ROOT, 
                            transform,
                            limit=40
                        ),
                        batch_size=1,
                        shuffle=False,
                        num_workers=4)

model = EncoderEstimator()
model.cuda()
# model = nn.DataParallel(model)

# compute loss and do backpropagation
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
end_epoch = 100

model.train()


for epoch in range(end_epoch):
    start = time.time()
    for i, data in enumerate(train_loader):
        loss = 0
        optimizer.zero_grad()

        (prev, kp_prev), (query, kp_query) = data
        prev = prev.cuda()
        query = query.cuda()

        for kp in range(len(kp_prev)):
            if kp_prev[kp] == 0:
                kp_prev[kp] = kp_prev[kp].cuda()
            else:
                kp_prev[kp][0] = kp_prev[kp][0]
                kp_prev[kp][1] = kp_prev[kp][1]
            
            if kp_query[kp] == 0:
                kp_query[kp] = kp_query[kp].cuda()
            else:
                kp_query[kp][0] = kp_query[kp][0].cuda()
                kp_query[kp][1] = kp_query[kp][1].cuda()

        prev_keys = model.get_keys(prev, kp_prev)
        query_keymap = model.encoder(query)
        
        kp_prev = model.downsample_annot(kp_prev)
        kp_query = model.downsample_annot(kp_query)

        for k in range(len(kp_query)):
            if kp_prev[k] == 0 or kp_query[k] == 0:
                continue

            dmap = model.get_dist_map(prev_keys[k], (kp_prev[k][0][0], kp_prev[k][1][0]), query_keymap, weighted=False)
            loss += dmap[kp_query[k][1][0]][kp_query[k][0][0]]

        
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

        # save checkpoint
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'loss': loss
        }, CHECKPOINT_PATH)

    writer.add_scalar("Train Loss", loss, epoch)
        