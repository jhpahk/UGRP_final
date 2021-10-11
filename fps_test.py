import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import cv2
from PIL import Image

import mmpose.datasets
from mmcv import Config
from torchvision.transforms.functional_pil import to_grayscale

from Lite_HRNet.models import build_posenet

from VideoPoseEstimationModel import VideoPoseEstimator

import pickle
import gzip

import time

# cfg = Config.fromfile("Lite_HRNet/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py")
# model_hrnet = build_posenet(cfg.model).cuda()
# model_hrnet.load_state_dict(torch.load("Lite_HRNet/litehrnet_18_mpii_256x256.pth")['state_dict'])
# if hasattr(model_hrnet, 'forward_dummy'):
#     model_hrnet.forward = model_hrnet.forward_dummy

models_encoder = []
for i in range(2, 10+1):
    models_encoder.append(VideoPoseEstimator(N=i, first=False).cuda())

# model_encoder = VideoPoseEstimator(N=5).cuda()

cap = cv2.VideoCapture("sample_videos/nwoo_jump.mp4")

total_times = torch.zeros(9)
# total_time = 0

frame_count = 0
# running_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((256, 256))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    # img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    # start = time.time()
    # model_encoder(img_tensor)
    # end = time.time()

    # total_time += (end - start)

    for i in range(9):
        start_time = time.time()
        models_encoder[i](img_tensor)
        end_time = time.time()

        total_times[i] += (end_time - start_time)

FPSs = frame_count / total_times
print(FPSs)

# FPS = frame_count / total_time
# print(FPS)


with gzip.open("fps_only_prev.pkl", "wb") as f:
    pickle.dump(FPSs, f)

# with gzip.open("test.pkl", "rb") as f:
#     data_test = pickle.load(f)

# print(data_test)