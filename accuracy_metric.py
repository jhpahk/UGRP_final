import torch

def joint_acc_metric(gt, target, return_mean = True):
    if return_mean:
        return torch.norm((gt-target), dim = 1).mean()
    else:
        return torch.norm((gt - target), dim = 1)

if __name__ == "__main__":
    gt =[[1,0], [3,4]]
    target =[[0, 0], [0, 0]]

    # gt = torch.tensor([[1,0], [3,4]], dtype = torch.float)
    # target = torch.tensor([[0, 0], [0, 0]], dtype = torch.float)
    print(torch.norm((gt - target), dim= 1).mean())
