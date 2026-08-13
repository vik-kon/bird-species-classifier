import torch
import torchvision

from torchvision import models

def get_model():
    model = torchvision.models.resnet18(weights='DEFAULT')


    
    model.fc = torch.nn.Linear(512,200)
            
    return model

