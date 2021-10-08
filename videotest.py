from VideoPoseEstimationModel import VideoPoseEstimator

from PIL import Image
from PIL import ImageDraw

import torchvision.transforms as transforms

import torch
import cv2
import numpy as np


def pil_draw_point(image, point):
    x, y = point
    draw = ImageDraw.Draw(image)
    radius = 2
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 0, 255))

    return image


def cv_draw_skeleton(img, joints):
    # 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax,
    # 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist

    blue = (255, 0, 0)

    if torch.is_tensor(joints[0][0]):
        new_joints = []
        for i in range(len(joints)):
            new_joints.append((joints[i][0].item(), joints[i][1].item()))

        joints = new_joints

    cv2.line(img, joints[0], joints[1], blue, 2)     # r ankle to r knee
    cv2.line(img, joints[1], joints[2], blue, 2)     # r knee to r hip
    cv2.line(img, joints[2], joints[6], blue, 2)     # r hip to pelvis
    cv2.line(img, joints[6], joints[3], blue, 2)     # pelvis to l hip
    cv2.line(img, joints[3], joints[4], blue, 2)     # l hip to l knee
    cv2.line(img, joints[4], joints[5], blue, 2)     # l knee to l ankle

    cv2.line(img, joints[6], joints[7], blue, 2)     # pelvis to thorax
    cv2.line(img, joints[7], joints[8], blue, 2)     # thorax to upper neck
    cv2.line(img, joints[8], joints[9], blue, 2)     # upper neck to head top

    cv2.line(img, joints[10], joints[11], blue, 2)   # r wrist to r elbow
    cv2.line(img, joints[11], joints[12], blue, 2)   # r elbow to r shoulder
    cv2.line(img, joints[12], joints[7], blue, 2)    # r shoulder to thorax
    cv2.line(img, joints[7], joints[13], blue, 2)    # thorax to l shoulder
    cv2.line(img, joints[13], joints[14], blue, 2)   # l shoulder to l elbow
    cv2.line(img, joints[14], joints[15], blue, 2)   # l elbow to l wrist

    return img


model = VideoPoseEstimator(N=2).cuda()

cap = cv2.VideoCapture("sample_joohyun.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)

    img = img.resize((256, 256))

    img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()

    joints = model(img_tensor)

    for i in range(len(joints)):
        if joints[i] == None:
            continue
        img = pil_draw_point(img, joints[i])

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv_draw_skeleton(img, joints)

    cv2.imshow('result', img)
    cv2.waitKey(1)