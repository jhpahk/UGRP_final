# from torchsummary import summary
# from model_autoencoder import VideoPoseEstimator

# model = VideoPoseEstimator().cuda()
# summary(model, (3, 256, 256))

import torch
from torch.nn.functional import softmax

t1 = torch.rand(2, 4, 4)
print(t1)

t2 = torch.rand(2, 4, 4)
print(t2)

distance = torch.norm(t2 - t1, dim=0)
print(distance)

distance = distance.reshape(16)
distance = softmax(distance, dim=0)
distance = distance.reshape(4, 4)

print(distance)