# from torchsummary import summary
# from model_autoencoder import VideoPoseEstimator

# model = VideoPoseEstimator().cuda()
# summary(model, (3, 256, 256))

import torch
import torch.nn.functional as F


def jaehyun():
    t1 = torch.rand(1, 2, 3, 3)
    k = torch.rand(1, 2, 1, 1)

    print(t1)
    result = F.conv2d(t1, k)
    print(t1)
    print(result)



def Nwoo():
    from torchvision.transforms.functional import crop
    t1 = torch.tensor([[0,0,0,0,0],
                       [0,1,1,1,0],
                       [0,1,1,1,0],
                       [0,1,1,1,0],
                       [0,0,0,0,0]], dtype= torch.float)
    t2 = crop(t1, 2 - 3//2,2 - 3//2,3,3)
    print(t2)
    

if __name__ == "__main__":
    # jaehyun()
    Nwoo()