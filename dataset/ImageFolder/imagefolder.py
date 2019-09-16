# coding=gbk
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

hymenoptera_dataset = datasets.ImageFolder(root='./test_images', transform=data_transform)
data_loader = DataLoader(hymenoptera_dataset, batch_size=6, shuffle=True)

for image, label in data_loader:
    print(image.size())
    print(label)
# .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif
