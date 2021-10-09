import torch
import torchvision.transforms as transforms

import os

from PIL import Image
from torch.utils.data import Dataset

class MPII(Dataset):
    def __init__(self):
        self.imglist = os.listdir("data/images/")[:10000]
        print(f"Successfully load {len(self.imglist)} images!")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = Image.open("data/images/" + self.imglist[idx]).convert("RGB")
        img = self.transform(img)

        return img