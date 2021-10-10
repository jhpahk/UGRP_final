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

from model_autoencoder_nwoo import EncoderEstimator
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

time_path = 'check_point_%04d_%02d_%02d_%02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
checkpoint_dir_path = "./checkpoint_nwoo/" + time_path
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

model = EncoderEstimator()

model.cuda()
model = nn.DataParallel(model)

# compute loss and do backpropagation
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
end_epoch = 40
criterion = nn.MSELoss()

model.train()
cnt = 0
for epoch in range(end_epoch):
    start = time.time()
    
    for i, data in enumerate(train_loader):
        print("{}th minibatch processing".format(i))

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

        prev_keys = model.module.get_keys(prev, kp_prev)
        query_keymap = model.module.encoder(query)

        kp_prev = model.module.downsample_annot(kp_prev)
        kp_query = model.module.downsample_annot(kp_query)
        # prev_keys = model.get_keys(prev, kp_prev)
        # query_keymap = model.encoder(query)[1]
        
        
        # kp_prev = model.downsample_annot(kp_prev)
        # kp_query = model.downsample_annot(kp_query)

        annot_maps = []
        distance_maps = []
        # make annotation map and distance map
        for k in range(len(prev_keys)):
            if kp_prev[k] == 0 or kp_query[k] == 0:
                continue
            
            # add kth annotation map
            kth_annot_map = 30 * torch.ones(1, 64, 64).cuda()
            kp_x = kp_query[k][0][0].item()
            kp_y = kp_query[k][1][0].item()

            kth_annot_map[0][kp_y][kp_x] = -1
            weight = -1

            left = False
            right = False
            up = False
            down = False
            
            for move in range(1, 64):
                weight += 0.3
                if kp_x - move >= 0:
                    kp_x_l = kp_x - move
                else:
                    kp_x_l = 0
                if kp_x + move < 64:
                    kp_x_r = kp_x + move
                else:
                    kp_x_r = 63

                if kp_y - move >= 0:
                    kp_y_u = kp_y - move
                else:
                    kp_y_u = 0
                if kp_y + move < 64:
                    kp_y_d = kp_y + move
                else:
                    kp_y_d = 63

                
                if not left:
                    kth_annot_map[:, kp_y_u:kp_y_d+1, kp_x_l] = weight
                    if kp_x_l == 0:
                        left = True
                if not right:
                    kth_annot_map[:, kp_y_u:kp_y_d+1, kp_x_r] = weight
                    if kp_x_r == 63:
                        right = True

                if not up:
                    kth_annot_map[:, kp_y_u, kp_x_l:kp_x_r+1] = weight
                    if kp_y_u == 0:
                        up = True
                if not down:
                    kth_annot_map[:, kp_y_d, kp_x_l:kp_x_r+1] = weight
                    if kp_y_d == 63:
                        down = True

            annot_maps.append(kth_annot_map)

            # add kth distance(prediction) map
            kth_dist_map = model.module.get_dist_map(prev_keys[k], kp_prev[k], query_keymap)
            distance_maps.append(kth_dist_map)
        
        if len(annot_maps) == 0:
            continue

        annot_maps = torch.stack(annot_maps)
        distance_maps = torch.stack(distance_maps)
        loss = torch.sum(annot_maps * distance_maps)

        # for k in range(len(prev_keys)):
        #     if kp_prev[k] == 0 or kp_query[k] == 0:
        #         continue

        #     kth_annot = kp_query[k]
        #     kth_dist_map = model.module.get_dist_map(prev_keys[k], query_keymap)
        #     loss += (1 - kth_dist_map[0][kth_annot[1][0]][[kth_annot[0][0]]])**2

        # if loss == 0:
        #     continue

        loss.backward()
        optimizer.step()
        if (cnt + 1) % 50 == 0:
        # save checkpoint
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'loss': loss
            }, CHECKPOINT_PATH)

            writer.add_scalar("Train Loss", loss, cnt // 50)
        cnt += 1


    end = time.time()
    time_cost = (end - start) / 60

    print(f"Epoch: {epoch + 1}/{end_epoch} -> Loss: {loss}, Time: {time_cost} min")

    # if (epoch + 1) % 10 == 0:
    #     # save checkpoint
    #     torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.module.state_dict(),
    #                 'optimizer_state_dict' : optimizer.state_dict(),
    #                 'loss': loss
    #     }, CHECKPOINT_PATH)

    # writer.add_scalar("Train Loss", loss, epoch)
        