import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16

def get_vit_b(num_classes):
    model = vit_b_16(pretrained=True)
    model.heads = nn.Linear(model.heads[0].in_features, num_classes)
    return model

def get_vit_l(num_classes):
    model = vit_l_16(pretrained=True)
    model.heads = nn.Linear(model.heads[0].in_features, num_classes)
    return model
