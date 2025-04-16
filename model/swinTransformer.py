import torch.nn as nn
from torchvision.models import swin_v2_b, swin_v2_s

def get_swin_v2_b(num_classes):
    model = swin_v2_b(pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def get_swin_v2_s(num_classes):
    model = swin_v2_s(pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
