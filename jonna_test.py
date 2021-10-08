from torchsummary import summary
from model_autoencoder import VideoPoseEstimator

model = VideoPoseEstimator().cuda()
summary(model, (3, 256, 256))