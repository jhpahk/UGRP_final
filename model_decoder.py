import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import crop, pad

from encoder import AutoEncoder

# input: 이전 frame의 keymap, annotation과 현재 frame의 keymap
# output: 현재 frame의 heatmap
class KeyPointDecoder(nn.Module):
    def __init__(self):
        super().__init__()


    def crop(self, img, center, size):
        x = center[0]
        y = center[1]

        width = size
        height = size
        half = (size - 1) // 2

        padding = [0, 0, 0, 0]     # Left, Top, Right, Bottom

        if x < half:
            width -= (half - x)
            padding[0] = size - width
            x = 0
        elif x >= (64 - half):
            width -= (half - (63 - x))
            padding[2] = size - width
            x -= half
        else:
            x -= half

        if y < half:
            height -= (half - y)
            padding[1] = size - height
            y = 0
        elif y >= (64 - half):
            height -= (half - (63 - y))
            padding[3] = size - height
            y -= half
        else:
            y -= half

        cropped = crop(img, y, x, height, width)
        cropped = pad(cropped, padding)

        return cropped


    def forward(self, x_prev, keypoints_prev, x_cur):
        heatmaps = []
        for i in range(len(keypoints_prev)):
            keypoint = keypoints_prev[i]
            if keypoint == torch.tensor([0]).cuda():
                continue

            kp_x = (keypoint[0].item() // 4)
            kp_y = (keypoint[1].item() // 4)
            kernel = self.crop(x_prev, (kp_x, kp_y), 3)

            heatmap = F.conv2d(x_cur, kernel, padding=1).squeeze()
            heatmaps.append(heatmap)
            print(torch.argmax(heatmap))
        
        if heatmaps == []:
            return None
            
        heatmaps = torch.stack(heatmaps)
        return heatmaps