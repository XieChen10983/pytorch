# coding=gbk
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


'-------------------------------------------------以下是自定义的dataset类的定义---------------------------------------------------'
# filenames是训练数据文件名称，labels是标签
class MyDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.files[idx]
        label = self.labels[idx]
        return image, label


class MyAnotherDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.files[idx]
        return image


'-------------------------------------------------------下面实现一个dataset类和一个dataloader----------------------------------------'
transform = transforms.Compose(
            [
                transforms.Scale((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
data_root = 'B:/machine learning/X光危险物品检测/data_set/mnist'
dataset = dict()
dataset['train'] = MNIST(root=data_root, train=True, download=True, transform=transform)
dataset['test'] = MNIST(root=data_root, train=False, download=True, transform=transform)
mydata = torch.randn(size=(60000, 3, 32, 32))
mylabel = torch.randint(low=0, high=9, size=(60000, ))
mydataset = MyDataset(files=mydata, labels=mylabel)
mydataloader = DataLoader(dataset=mydataset, batch_size=32, shuffle=True, num_workers=0)
# a = Dataset()
myanotherdataset = MyAnotherDataset(dataset['train'].train_data)
myanotherdataloader = DataLoader(myanotherdataset, batch_size=100, shuffle=True)

for image in myanotherdataloader:
    print(image.size())
