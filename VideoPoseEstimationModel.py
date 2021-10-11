import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import copy

import torchvision.transforms as transforms
from torchvision.transforms.functional import crop

from PIL import Image

import mmpose.datasets
from mmcv import Config

from model_autoencoder import EncoderEstimator
# from nwoo.model_autoencoder_nwoo import EncoderEstimator
from Lite_HRNet.models import build_posenet


class VideoPoseEstimator(nn.Module):
    def __init__(self, N=10, first=True, pretrained="autoencoder/UGRP_AutoEncoder/checkpoint/checkpoint_2021_10_09_04:42:23/checkpoint_2021_10_09_04:42:23_epoch_100.pth"):
        super().__init__()
        self.N = N
        self.first = first
        self.model_encoder = EncoderEstimator(pretrained)
        # self.model_encoder.load_state_dict(torch.load(pretrained)['model_state_dict'])

        cfg = Config.fromfile("Lite_HRNet/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py")
        self.model_hrnet = build_posenet(cfg.model)
        self.model_hrnet.load_state_dict(torch.load("Lite_HRNet/litehrnet_18_mpii_256x256.pth")['state_dict'])
        if hasattr(self.model_hrnet, 'forward_dummy'):
            self.model_hrnet.forward = self.model_hrnet.forward_dummy

        self.count = 0

        self.prev_img = None
        self.prev_result = None
        self.prev_keymap = None
        self.prev_keys = None

        self.first_img = None
        self.first_hrnet_result = None
        self.first_keymap = None
        self.first_keys = None


    def crop(self, img, center, size):
        x = center[0]
        y = center[1]

        target_x = 2
        target_y = 2

        width = size
        height = size
        half = (size - 1) // 2

        if x < half:
            width -= (half - x)
            target_x -= (half - x)
            x = 0
        elif x >= (64 - half):
            width -= (half - (63 - x))
            x -= half
        else:
            x -= half

        if y < half:
            height -= (half - y)
            target_y -= (half - y)
            y = 0
        elif y >= (64 - half):
            height -= (half - (63 - y))
            y -= half
        else:
            y -= half

        cropped = crop(img, y, x, height, width)
        return cropped, (target_x, target_y)


    def weighting(self, img, center):
        x, y = center

        left = False
        right = False
        up = False
        down = False

        weight = 0.5
        weight_map = weight * torch.ones_like(img)
        height, width = weight_map.shape[0], weight_map.shape[1]

        for move in range(1, 2 + 1):
            weight += 0.1
            if x - move >= 0:
                x_l = x - move
            else:
                x_l = 0
            if x + move < width:
                x_r = x + move
            else:
                x_r = width - 1

            if y - move >= 0:
                y_u = y - move
            else:
                y_u = 0
            if y + move < height:
                y_d = y + move
            else:
                y_d = height - 1
            
            if not left:
                weight_map[y_u:y_d+1, x_l] = weight
                if x_l == 0:
                    left = True
            if not right:
                weight_map[y_u:y_d+1, x_r] = weight
                if x_r == width - 1:
                    right = True

            if not up:
                weight_map[y_u, x_l:x_r+1] = weight
                if y_u == 0:
                    up = True
            if not down:
                weight_map[y_d, x_l:x_r+1] = weight
                if y_d == height - 1:
                    down = True
        
        return img * weight_map


    def forward(self, img):
        if self.count % self.N == 0:
            # print("Lite-HRNet")
            self.first_img = img
            self.prev_img = img
            keymaps = self.model_hrnet(img)

            keypoints = []
            for i in range(keymaps.shape[1]):
                idx_max = torch.argmax(keymaps[0][i])
                x = idx_max % 64
                y = idx_max // 64
                keypoints.append((x, y))

            # keypoints = []
            # for i in range(keymaps.shape[1]):
            #     keymap = keymaps[0][i].unsqueeze(0).unsqueeze(0)
            #     keymap = F.interpolate(keymap, scale_factor=4, mode="bilinear")
            #     idx_max = torch.argmax(keymap)
            #     x = idx_max % 256
            #     y = idx_max // 256
            #     keypoints.append((x, y))
            
            
            self.first_hrnet_result = keypoints
            self.prev_result = self.first_hrnet_result
            # print(self.prev_result)

            if self.first:
                self.first_keymap = self.model_encoder.encoder(self.first_img)
                self.first_keys = self.model_encoder.get_keys_with_keymap(self.first_keymap, keypoints)

                self.prev_keymap = self.first_keymap
                self.prev_keys = self.first_keys

            self.count += 1
            return keypoints

        else:
            # print("My Encoder Model")
            if self.first:
                query_keymap = self.model_encoder.encoder(img)
                # print(self.prev_result)
                result = []
                for k in range(len(self.prev_keys)):
                    prev_kp = copy.deepcopy(self.prev_result)
                    # print(self.prev_result[k])
                    keymap_cropped, target = self.crop(query_keymap, prev_kp[k], 5)
                    # print("1", self.prev_result[k])
                    prev_x, prev_y = self.prev_result[k][0], self.prev_result[k][1]
                    # print("2", prev_x, prev_y)

                    dmap_first = self.model_encoder.get_dist_map(self.first_keys[k], keymap_cropped).squeeze()
                    dmap_first = self.weighting(dmap_first, target)
                    # print("3", prev_x, prev_y)

                    dmap_prev = self.model_encoder.get_dist_map(self.prev_keys[k], keymap_cropped).squeeze()
                    dmap_prev = self.weighting(dmap_prev, target)
                    # print("4", prev_x, prev_y)

                    dmap = 0.25 * dmap_first + 0.75 * dmap_prev
                    # print("5", prev_x, prev_y)

                    min_idx = torch.argmin(dmap)
                    min_x = min_idx % dmap_prev.shape[1]
                    min_y = min_idx // dmap_prev.shape[1]
                    x = prev_x + (min_x - target[0])
                    y = prev_y + (min_y - target[1])
                    # print((prev_x, prev_y), (x, y))

                    # map_64 = torch.zeros(1, 1, 64, 64)
                    # map_64[0][0][y_64][x_64] = 1
                    # map_256 = F.interpolate(map_64, scale_factor=4, mode="bilinear")
                    # target_idx = torch.argmax(map_256)
                    # x = target_idx % 256
                    # y = target_idx // 256

                    result.append((x, y))


                self.prev_img = img
                self.prev_result = result
                self.prev_keymap = query_keymap
                self.prev_keys = self.model_encoder.get_keys_with_keymap(query_keymap, result)
                self.count += 1

                return result

            else:
                if self.count % self.N == 1:
                    self.model_encoder.set_memory(self.first_img, self.first_hrnet_result)

                self.count += 1          
                return self.model_encoder(img)


if __name__ == "__main__":
    img = Image.open("data/images/test/000005_mpiinew_test/000000.jpg")
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img).unsqueeze(0)

    model = VideoPoseEstimator()
    print(model(img))