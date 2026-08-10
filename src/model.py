import torch
import torchvision

from torchvision import models

def get_model():
    model = torchvision.models.resnet18(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    
    model.fc = torch.nn.Linear(512,200)
            
    return model

