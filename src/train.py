import torch
import torch.nn as nn
from model import get_model
from dataset import train_loader 


model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch{epoch + 1} loss: {loss.item()}")

torch.save(model.state_dict(), "models/model.pth")