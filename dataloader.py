from __future__ import print_function, division
import glob
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        # transforms.RandomRotation(degrees=30, expand=True), 
        # transforms.RandomResizedCrop(224, scale=(0.45, 1.0), ratio=(3./4, 4./3)),
        # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(5, scale=[0.95, 1.05]),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class XunFeiDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        # print(self.img_path[index], cv2.imread(self.img_path[index]).shape)
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.img_label[index]
        return img, label

    def __len__(self):
        return len(self.img_path)


def get_dataloader(args, test=False):

    train_df = pd.DataFrame({'path': glob.glob('./data/train/*/*')})
    train_df['label'] = train_df['path'].apply(lambda x: int(x.split('/')[-2]))
    # train_df['label'].nunique()
    train_df = train_df.sample(frac=1.0)
    cls_count = train_df['label'].value_counts()
    cls_count = [cls_count[i] for i in range(len(cls_count))]

    train_size = int((1-args.val_rate) * len(train_df['label']))

    train_loader = DataLoaderX(
                        XunFeiDataset(train_df['path'].values[:train_size], 
                            train_df['label'].values[:train_size],
                            transform=train_transform),
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        num_workers=args.num_workers,
                        pin_memory=True)
    
    val_loader = DataLoaderX(
                    XunFeiDataset(train_df['path'].values[train_size:], 
                        train_df['label'].values[train_size:],
                        transform=train_transform), 
                    batch_size=args.test_batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers,
                    pin_memory=True)
    
    test_df = pd.DataFrame({'path': glob.glob('../data/test/*')})
    test_df['label'] = 0
    test_loader = DataLoaderX(
                    XunFeiDataset(test_df['path'].values[:], 
                        test_df['label'].values[:],
                        transform=test_transform), 
                    batch_size=args.test_batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers,
                    pin_memory=True)

    if test:
        return test_loader

    return train_loader, val_loader, cls_count


