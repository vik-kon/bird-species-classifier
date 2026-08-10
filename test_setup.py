import torch
import torchvision

print(torch.__version__)

print(torch.cuda.is_available())

print(torchvision.models.resnet18(weights='DEFAULT'))