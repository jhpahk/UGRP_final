import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import softmax
from torchvision.transforms.functional import crop

from encoder import AutoEncoder

class EncoderEstimator(nn.Module):
    def __init__(self, pretrained="autoencoder/UGRP_AutoEncoder/checkpoint/checkpoint_2021_10_09_04:42:23/checkpoint_2021_10_09_04:42:23_epoch_70.pth"):
        super().__init__()
        self.autoencoder = AutoEncoder()
        self.autoencoder.load_state_dict(torch.load(pretrained)['model_state_dict'])
        self.encoder = self.autoencoder.encoder

        self.memory_keys = []
        self.memory_joints = []

        self.threshold = 5


    def downsample_annot(self, annot):      # 256x256 resolution에서의 annotation을 64x64에 맞게 downsampling한다.
        annot_downsampled = []
        for i in range(len(annot)):
            if annot[i] == 0:
                annot_downsampled.append(0)
            else:
                annot_downsampled.append(((annot[i][1] // 4), (annot[i][0] // 4)))

        return annot_downsampled


    def get_keys(self, frame, annot):        # frame image(256x256)와 annotation(256x256)을 주면, 관절에 해당하는 key vectors를 뽑아준다.
        keymap = self.encoder(frame)[1]
        annot = self.downsample_annot(annot)

        keys = []
        for i in range(len(annot)):
            if annot[i] == 0:
                # keys.append(torch.zeros(1, 32, 1, 1).cuda())
                keys.append(torch.zeros(1, 32, 1, 1))
            else:
                # keys.append(crop(keymap, annot[i][1], annot[i][0], 1, 1).cuda())
                keys.append(crop(keymap, annot[i][1], annot[i][0], 1, 1))
        
        return keys


    def set_memory(self, frame, annot):
        self.memory_keys = self.get_keys(frame, annot)
        self.memory_joints = self.downsample_annot(annot)


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


    # target key vector와 query key map이 주어지면,
    # query key map의 좌표별 key vector와 target key vector 사이의 eudlidean distance를 계산하여
    # distance matrix를 만들어 리턴한다.
    def calc_distance(self, target, target_point, query, weighted=True):
        distance = torch.norm((query - target), dim=1)
        if weighted:
            weight_map = self.make_weight_map(target_point)
            distance = distance * weight_map

        return distance


    def get_dist_map(self, target, target_point, query):
        distance_map = self.calc_distance(target, target_point, query, weighted=False)
        distance_map = distance_map.reshape(4096)
        distance_map = softmax(distance_map, dim=0)
        distance_map = distance_map.reshape(1, 64, 64)

        return distance_map


    def forward(self, query):
        keymap_query = self.encoder(query)[1]      # query image로부터 query key map을 생성한다.
        joints = []
        for i in range(len(self.memory_keys)):       # 각각의 joint에 대해 !!! memory[i]가 0일 경우 추가!
            # if torch.all(self.memory_keys[i] == torch.zeros(1, 32, 1, 1).cuda()):
            if torch.all(self.memory_keys[i] == torch.zeros(1, 32, 1, 1)):
                joints.append(None)
                continue

            distance = self.calc_distance(self.memory_keys[i], self.memory_joints[i], keymap_query)     # query key map과의 distance를 계산한 뒤,
            if (int(torch.min(distance)) > self.threshold):                 # 만약 가장 유사한 key와의 distance가 미리 정의한 threshold를 넘으면
                joints.append(None)
                continue                                                    # 해당 관절이 query에 존재하지 않는다고 판단한다.

            min_idx = int(torch.argmin(distance))       # distance가 최소인 key vector의 index를 구하고,
            min_vertical = min_idx // 64                # index를 기반으로 vertical coordinate와
            min_horizontal = min_idx % 64               # horizontal coordinate를 구한다.

            self.memory_keys[i] = crop(keymap_query, min_vertical, min_horizontal, 1, 1)     # 해당 joint에 대한 memory를 query key 값으로 대체한다.
            self.memory_joints[i] = (min_horizontal, min_vertical)

            # distance map을 x4 upsampling하여 original resolution을 회복시킨 뒤, 가장 distance가 작은 좌표를 joint로 지정한다.
            distance = distance.unsqueeze(0)
            distance_x4 = F.interpolate(distance, scale_factor=4, mode='bilinear')
            scaled_min_idx = int(torch.argmin(distance_x4))
            scaled_min_vertical = scaled_min_idx // 256
            scaled_min_horizontal = scaled_min_idx % 256
            joints.append((scaled_min_vertical, scaled_min_horizontal))
        
        return joints