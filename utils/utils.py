import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image

def accuracy(output, target):
    """����׼ȷ��"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)

def generate_gradcam(model, image, target_layer, class_idx=None):
    """
    ���� Grad-CAM ����ͼ
    Args:
        model (nn.Module): ģ��
        image (torch.Tensor): ����ͼ�� (C, H, W)
        target_layer (str): Ŀ����������
        class_idx (int, optional): ��������
    Returns:
        heatmap (np.ndarray): ����ͼ
    """
    model.eval()

    # ��ȡĿ��������ͼ
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output

    feature_map = None
    hook = dict(model.named_modules())[target_layer].register_forward_hook(forward_hook)

    # ǰ�򴫲�
    image = image.unsqueeze(0)  # ��� batch ά��
    output = model(image)
    if class_idx is None:
        class_idx = output.argmax().item()

    # ���� Grad-CAM Ȩ��
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1
    output.backward(gradient=one_hot)

    gradients = model.get_gradients()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_map).sum(dim=1).squeeze()

    # ��������ͼ
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))
    cam = cam / cam.max()

    hook.remove()
    return cam
