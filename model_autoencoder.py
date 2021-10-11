import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from torch.nn.functional import softmax
from torch.nn.modules import distance
from torchvision.transforms.functional import crop

from encoder import AutoEncoder

class EncoderEstimator(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.autoencoder = AutoEncoder()
        self.autoencoder.load_state_dict(torch.load(pretrained)['model_state_dict'])
        self.encoder = self.autoencoder.encoder
        
        self.memory_keys = []
        self.memory_keypoints = []

        self.threshold = 5


    # def downsample_annot(self, annot):      # 256x256 resolution에서의 annotation을 64x64에 맞게 downsampling한다.
    #     annot_downsampled = []
    #     for i in range(len(annot)):
    #         if annot[i] == 0:
    #             annot_downsampled.append(0)
    #         else:
    #             annot_downsampled.append(((annot[i][0] // 4), (annot[i][1] // 4)))

    #     return annot_downsampled


    def get_keys(self, frame, annot):        # frame image(256x256)와 annotation(64x64)을 주면, 관절에 해당하는 key vectors를 뽑아준다.
        keymap = self.encoder(frame)
        # annot = self.downsample_annot(annot)

        keys = []
        for i in range(len(annot)):
            if annot[i] == 0:
                keys.append(torch.zeros(1, 32, 1, 1).cuda())
                # keys.append(torch.zeros(1, 32, 1, 1))
            else:
                keys.append(crop(keymap, annot[i][1], annot[i][0], 1, 1).cuda())
                # keys.append(crop(keymap, annot[i][1], annot[i][0], 1, 1))
        
        return keys
    

    def get_keys_with_keymap(self, keymap, annot):
        # annot = self.downsample_annot(annot)
        keys = []
        for i in range(len(annot)):
            keys.append(crop(keymap, annot[i][1], annot[i][0], 1, 1).cuda())
            # keys.append(crop(keymap, annot[i][1], annot[i][0], 1, 1))
        
        return keys


    def set_memory(self, frame, annot):
        self.memory_keys = self.get_keys(frame, annot)
        # self.memory_keypoints = self.downsample_annot(annot)
        self.memory_keypoints = annot


    # target key vector와 query key map이 주어지면,
    # query key map의 좌표별 key vector와 target key vector 사이의 eudlidean distance를 계산하여
    # distance map을 만들어 리턴한다.
    def calc_distance(self, target, query):
        distance = torch.norm((query - target), dim=1)
        return distance


    def get_dist_map(self, target, query):
        distance_map = self.calc_distance(target, query)
        shape = distance_map.shape
        distance_map = distance_map.reshape(shape[1] * shape[2])
        distance_map = softmax(distance_map, dim=0)
        distance_map = distance_map.reshape(shape)

        return distance_map


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
            x = 0
            target_x -= (half - x)
        elif x >= (64 - half):
            width -= (half - (63 - x))
            x -= half
        else:
            x -= half

        if y < half:
            height -= (half - y)
            y = 0
            target_y -= (half - y)
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


    def forward(self, query):
        keymap_query = self.encoder(query)
        keypoints = []
        for i in range(len(self.memory_keys)):
            prev_kp = copy.deepcopy(self.memory_keypoints)
            keymap_cropped, target = self.crop(keymap_query, prev_kp[i], 5)
            prev_x, prev_y = self.memory_keypoints[i][0], self.memory_keypoints[i][1]
            distance = self.calc_distance(self.memory_keys[i], keymap_cropped).squeeze()
            distance = self.weighting(distance, target)

            min_idx = torch.argmin(distance)
            min_x = min_idx % 5
            min_y = min_idx // 5
            move_x = target[0] - min_x
            move_y = target[1] - min_y

            x = prev_x + move_x
            y = prev_y + move_y

            self.memory_keys[i] = crop(keymap_query, y, x, 1, 1)
            self.memory_keypoints[i] = (x, y)

            keypoints.append((x, y))
        
        return keypoints