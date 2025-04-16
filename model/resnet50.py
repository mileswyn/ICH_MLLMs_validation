import torch.nn as nn
from torchvision.models import resnet50

def get_resnet50(num_classes):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
