from VideoPoseEstimationModel import VideoPoseEstimator

from PIL import Image
from PIL import ImageDraw

import torchvision.transforms as transforms

import cv2
import numpy as np


def pil_draw_point(image, point):
    x, y = point
    draw = ImageDraw.Draw(image)
    radius = 2
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 255))

    return image


model = VideoPoseEstimator(N=3)

cap = cv2.VideoCapture("sample_joohyun.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    img = img.resize((256, 256))

    img_tensor = transforms.ToTensor()(img).unsqueeze(0)

    joints = model(img_tensor)

    for i in range(len(joints)):
        if joints[i] == None:
            continue
        img = pil_draw_point(img, joints[i])

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    cv2.imshow('result', img)