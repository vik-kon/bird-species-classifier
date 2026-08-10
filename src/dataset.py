import torch, os
from torchvision import transforms, datasets

from torch.utils.data import DataLoader
from PIL import Image


data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    ),
])

class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, train):
        super().__init__()

        #images 
        images = {}
        with open(os.path.join(root, "images.txt")) as f:
            for line in f:
                image_id, path = line.strip().split()
                images[image_id] = path

        #labels
        labels = {}
        with open(os.path.join(root ,"image_class_labels.txt")) as f:
            for line in f:
                image_id, path = line.strip().split()
                labels[image_id] = path

        splits = {}
        with open(os.path.join(root, "train_test_split.txt")) as f:
                    for line in f:
                        image_id, path = line.strip().split()
                        splits[image_id] = path

        self.transform = transform
        self.samples = []
        for image_id in images:
             if int(splits[image_id]) == (1 if train else 0):
                  path = os.path.join(root, "images", images[image_id])
                  label = int(labels[image_id]) - 1
                  self.samples.append((path,label))

        

    def __len__(self):
         return len(self.samples)

    def __getitem__(self, idx):
         path, label = self.samples[idx]
         image = Image.open(path).convert("RGB")
         if self.transform:
              image = self.transform(image)
         return image, label


train_dataset = CUBDataset(root = "data/CUB_200_2011", transform = data_transforms, train = True)

train_loader = DataLoader(train_dataset, batch_size= 32, shuffle = True)

test_dataset = CUBDataset(root = "data/CUB_200_2011", transform = data_transforms, train = False)

test_loader = DataLoader(test_dataset, batch_size= 32, shuffle = False)


