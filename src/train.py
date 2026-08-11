import torch
import torch.nn as nn
from model import get_model
from dataset import train_loader, test_loader 


model = get_model()
optimizer = torch.optim.Adam(model.fc.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch{epoch + 1} loss: {loss.item()}")

