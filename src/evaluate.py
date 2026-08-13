import torch
from model import get_model
from dataset import test_loader

model = get_model()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        #get output
        output = model(images)
        # get predicted class using torch.argmax
        predicted_class = torch.argmax(output, dim = 1)
        #update correct and total
        correct += (predicted_class == labels).sum().item()
        total += labels.size(0)


print(f"Test Accuracy: {100 * correct/total:.2f}%")