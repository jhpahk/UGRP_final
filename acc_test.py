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
from accuracy_metric import joint_acc_metric

import pickle
import gzip

cfg = Config.fromfile("Lite_HRNet/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py")
model_hrnet = build_posenet(cfg.model).cuda()
model_hrnet.load_state_dict(torch.load("Lite_HRNet/litehrnet_18_mpii_256x256.pth")['state_dict'])
if hasattr(model_hrnet, 'forward_dummy'):
    model_hrnet.forward = model_hrnet.forward_dummy

models_encoder = []
for i in range(2, 10+1):
    models_encoder.append(VideoPoseEstimator(N=i, first=True).cuda())

cap = cv2.VideoCapture("sample_videos/nwoo_jump.mp4")

frame_counts = torch.zeros(9)
total_norms = torch.zeros(9)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((256, 256))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()
    
    hrnet_heatmap = model_hrnet(img_tensor)
    result_hrnet = []
    for i in range(hrnet_heatmap.shape[1]):
        keymap = hrnet_heatmap[0][i]
        idx_max = torch.argmax(keymap)
        x = idx_max % 64
        y = idx_max // 64
        result_hrnet.append((x, y))
    result_hrnet = torch.tensor(result_hrnet, dtype=torch.float)

    for i in range(9):
        results = torch.tensor(models_encoder[i](img_tensor), dtype=torch.float)
        norm = joint_acc_metric(result_hrnet, results)
        if norm != 0:
            total_norms[i] += norm
            frame_counts[i] += 1

avg_norms = total_norms / frame_counts
torch.set_printoptions(precision=10)
print(avg_norms)


with gzip.open("acc_first_prev.pkl", "wb") as f:
    pickle.dump(avg_norms, f)

# with gzip.open("test.pkl", "rb") as f:
#     data_test = pickle.load(f)

# print(data_test)