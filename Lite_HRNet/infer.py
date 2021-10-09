import torch
import mmpose.datasets

from models import build_posenet
from mmcv import Config

cfg = Config.fromfile("configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py")
model = build_posenet(cfg.model)
if hasattr(model, 'forward_dummy'):
    model.forward = model.forward_dummy

example = torch.rand(1, 3, 64, 64)
torch.set_printoptions(profile="full", precision=5, sci_mode=False)
print(model(example))