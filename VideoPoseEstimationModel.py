import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

from PIL import Image

import mmpose.datasets
from mmcv import Config

from model_autoencoder import EncoderEstimator
from Lite_HRNet.models import build_posenet


class VideoPoseEstimator(nn.Module):
    def __init__(self, N=10):
        super().__init__()
        self.N = N
        self.model_encoder = EncoderEstimator()

        cfg = Config.fromfile("Lite_HRNet/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py")
        self.model_hrnet = build_posenet(cfg.model)
        self.model_hrnet.load_state_dict(torch.load("Lite_HRNet/litehrnet_18_mpii_256x256.pth")['state_dict'])
        if hasattr(self.model_hrnet, 'forward_dummy'):
            self.model_hrnet.forward = self.model_hrnet.forward_dummy

        self.count = 0

        self.prev_img = None
        self.prev_hrnet_result = []


    def forward(self, img):
        if self.count % self.N == 0:
            print("Lite-HRNet")
            self.prev_img = img
            keymaps = self.model_hrnet(img)

            joints = []
            for i in range(keymaps.shape[1]):
                keymap = keymaps[0][i].unsqueeze(0).unsqueeze(0)
                keymap = F.interpolate(keymap, scale_factor=4, mode="bilinear")
                idx_max = torch.argmax(keymap)
                x = idx_max % 256
                y = idx_max // 256
                joints.append((x, y))
            
            self.prev_hrnet_result = joints
            self.count += 1
            return joints

        else:
            print("My Encoder Model")
            if self.count % self.N == 1:
                self.model_encoder.set_memory(self.prev_img, self.prev_hrnet_result)
            self.count += 1
            return self.model_encoder(img)


if __name__ == "__main__":
    img = Image.open("data/images/test/000005_mpiinew_test/000000.jpg")
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img).unsqueeze(0)

    model = VideoPoseEstimator()
    print(model(img))