import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import os
import torch


def process_label(label_array, num_classes=5):
    """
    将像素级标签图像转换为多标签向量。
    
    Args:
        label_path (str): 标签图像的路径（.png 文件）。
        num_classes (int): 类别数（1-5）。
        
    Returns:
        torch.Tensor: 长度为 num_classes 的多标签向量。
    """

    # 初始化多标签向量
    multi_label = np.zeros(num_classes, dtype=np.float32)
    # 检查每个类别是否存在
    # for i in range(1, num_classes + 1):  # 类别从 1 到 5
    if 51 in label_array:
        multi_label[0] = 1.0  # 如果类别 i 存在，设置为 1
    if 102 in label_array:
        multi_label[1] = 1.0
    if 153 in label_array:        
        multi_label[2] = 1.0
    if 204 in label_array:    
        multi_label[3] = 1.0        
    if 255 in label_array:
        multi_label[4] = 1.0

    # print(np.unique(label_array).tolist())
    
    return multi_label

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
    def __init__(self, data_dir, mode="train", transform=None, num_classes=5):
        """
        Args:
            image_dir (str): 图像文件夹路径。
            label_dir (str): 标签文件夹路径（.png 文件）。
            transform (callable, optional): 图像预处理变换。
            num_classes (int): 类别数（1-5）。
        """
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.image_dir = os.path.join(data_dir, f"{mode}/images")
        self.label_dir = os.path.join(data_dir, f"{mode}/labels")
        
        # 获取所有图像文件名
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

        assert len(self.image_files) == len(self.label_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # 加载标签
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        label = Image.open(label_path)
        label = np.array(label)
        if label.ndim == 3:
            label = label[:, :, 0]
        label = process_label(label, num_classes=self.num_classes)
        
        # 应用预处理
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
