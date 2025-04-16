# -*- encoding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class HemorrhageDataset(Dataset):
    def __init__(self, data_dir, mode="train", transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.image_dir = os.path.join(data_dir, f"{mode}/images")
        self.label_dir = os.path.join(data_dir, f"{mode}/labels")

        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        assert len(self.image_files) == len(self.label_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = Image.open(label_path)

        label = np.array(label)
        if label.ndim == 3:
            label = label[:, :, 0]

        hemorrhage = 1 if np.any(label > 0) else 0

        if self.transform:
            image = self.transform(image)

        return image, hemorrhage
