import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import softmax
from torchvision.transforms.functional import crop

from encoder import AutoEncoder
from model_decoder import KeyPointDecoder

class EncoderEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = AutoEncoder()
        self.autoencoder.load_state_dict(torch.load("autoencoder/UGRP_AutoEncoder/checkpoint/checkpoint_2021_10_09_04:42:23/checkpoint_2021_10_09_04:42:23_epoch_100.pth")['model_state_dict'])
        self.encoder = self.autoencoder.encoder

        self.decoder = KeyPointDecoder()

        self.prev_keymap = None
        self.prev_joints = []

        self.threshold = 5


    def downsample_annot(self, annot):      # 256x256 resolution에서의 annotation을 64x64에 맞게 downsampling한다.
        annot_downsampled = []
        for i in range(len(annot)):
            if annot[i] == 0:
                annot_downsampled.append(0)
            else:
                annot_downsampled.append(((annot[i][0] // 4), (annot[i][1] // 4)))

        return annot_downsampled


    def set_memory(self, keymap, annot):
        self.prev_keymap = keymap
        self.prev_joints = annot


    def make_weight_map(self, keypoint):
        # weight_map = 10 * torch.ones(1, 64, 64).cuda()
        weight_map = 10 * torch.ones(1, 64, 64)
        kp_x, kp_y = keypoint
        weight = 0.5

        left = False
        right = False
        up = False
        down = False
        
        for move in range(1, 16):
            weight += 0.25
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
                weight_map[:, kp_y_u:kp_y_d+1, kp_x_l] = weight
                if kp_x_l == 0:
                    left = True
            if not right:
                weight_map[:, kp_y_u:kp_y_d+1, kp_x_r] = weight
                if kp_x_r == 63:
                    right = True

            if not up:
                weight_map[:, kp_y_u, kp_x_l:kp_x_r+1] = weight
                if kp_y_u == 0:
                    up = True
            if not down:
                weight_map[:, kp_y_d, kp_x_l:kp_x_r+1] = weight
                if kp_y_d == 63:
                    down = True

        return weight_map


    def forward(self, query):
        keymap_query = self.encoder(query)
        heatmaps = self.decoder(self.prev_keymap, self.prev_joints, keymap_query)
        keypoints = []
        
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i]
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)
            heatmap = F.interpolate(heatmap, scale_factor=4, mode='bilinear')
            heatmap = heatmap.squeeze()

            min_idx = int(torch.argmax(heatmap))
            min_vertical = min_idx // 256
            min_horizontal = min_idx % 256
            keypoints.append((min_horizontal, min_vertical))

        self.prev_keymap = keymap_query
        self.prev_joints = keypoints
        return keypoints