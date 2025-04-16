import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image

def accuracy(output, target):
    """计算准确率"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)

def generate_gradcam(model, image, target_layer, class_idx=None):
    """
    生成 Grad-CAM 热力图
    Args:
        model (nn.Module): 模型
        image (torch.Tensor): 输入图像 (C, H, W)
        target_layer (str): 目标卷积层名称
        class_idx (int, optional): 分类索引
    Returns:
        heatmap (np.ndarray): 热力图
    """
    model.eval()

    # 获取目标层的特征图
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output

    feature_map = None
    hook = dict(model.named_modules())[target_layer].register_forward_hook(forward_hook)

    # 前向传播
    image = image.unsqueeze(0)  # 添加 batch 维度
    output = model(image)
    if class_idx is None:
        class_idx = output.argmax().item()

    # 计算 Grad-CAM 权重
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)

    gradients = model.get_gradients()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_map).sum(dim=1).squeeze()

    # 生成热力图
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam = cam / cam.max()

    hook.remove()
    return cam
