import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from dataset.dataset_multi import HemorrhageDataset, get_transforms
from model.resnet50 import get_resnet50
from model.vit import get_vit_b, get_vit_l
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "resnet50"
model_path = "./best_model_multi_resnet50.pth"
data_dir = "./data"
batch_size = 1
output_file = "./test_results_resnet50_multi.txt"

test_dataset = HemorrhageDataset(data_dir=data_dir, mode='test', transform=get_transforms('test'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if model_name == "resnet50":
    model = get_resnet50(num_classes=5).to(device)
elif model_name == "vit":
    model = get_vit_b(num_classes=5).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_labels = []
all_preds = []
test_edh_correct, test_iph_correct, test_ivh_correct, test_sah_correct, test_sdh_correct = 0, 0, 0, 0, 0
test_total = 0
edh_all_labels, edh_all_preds = [], []
iph_all_labels, iph_all_preds = [], []
ivh_all_labels, ivh_all_preds = [], [] 
sah_all_labels, sah_all_preds = [], []
sdh_all_labels, sdh_all_preds = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        predicted = torch.sigmoid(outputs) > 0.5
        test_total += labels.size(0)

        test_edh_correct += predicted[:, 0].eq(labels[:, 0]).sum().item()
        test_iph_correct += predicted[:, 1].eq(labels[:, 1]).sum().item()
        test_ivh_correct += predicted[:, 2].eq(labels[:, 2]).sum().item()
        test_sah_correct += predicted[:, 3].eq(labels[:, 3]).sum().item()
        test_sdh_correct += predicted[:, 4].eq(labels[:, 4]).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        edh_all_labels.extend(labels[:, 0].cpu().numpy())
        edh_all_preds.extend(predicted[:, 0].cpu().numpy())
        iph_all_labels.extend(labels[:, 1].cpu().numpy())
        iph_all_preds.extend(predicted[:, 1].cpu().numpy()) 
        ivh_all_labels.extend(labels[:, 2].cpu().numpy())
        ivh_all_preds.extend(predicted[:, 2].cpu().numpy())
        sah_all_labels.extend(labels[:, 3].cpu().numpy())
        sah_all_preds.extend(predicted[:, 3].cpu().numpy())
        sdh_all_labels.extend(labels[:, 4].cpu().numpy())
        sdh_all_preds.extend(predicted[:, 4].cpu().numpy())

test_edh_acc = 100. * test_edh_correct / test_total
test_iph_acc = 100. * test_iph_correct / test_total
test_ivh_acc = 100. * test_ivh_correct / test_total
test_sah_acc = 100. * test_sah_correct / test_total
test_sdh_acc = 100. * test_sdh_correct / test_total
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average=None)
recall = recall_score(all_labels, all_preds, average=None)
f1 = f1_score(all_labels, all_preds, average=None)
edh_tn, edh_fp, edh_fn, edh_tp = confusion_matrix(edh_all_labels, edh_all_preds).ravel()
# 计算特异性
edh_specificity = edh_tn / (edh_tn + edh_fp)
print("EDH Specificity: ", edh_specificity)
iph_tn, iph_fp, iph_fn, iph_tp = confusion_matrix(iph_all_labels, iph_all_preds).ravel()
# 计算特异性
iph_specificity = iph_tn / (iph_tn + iph_fp)
print("IPH Specificity: ", iph_specificity)
ivh_tn, ivh_fp, ivh_fn, ivh_tp = confusion_matrix(ivh_all_labels, ivh_all_preds).ravel()
# 计算特异性
ivh_specificity = ivh_tn / (ivh_tn + ivh_fp)
print("IVH Specificity: ", ivh_specificity)
sah_tn, sah_fp, sah_fn, sah_tp = confusion_matrix(sah_all_labels, sah_all_preds).ravel()
# 计算特异性
sah_specificity = sah_tn / (sah_tn + sah_fp)
print("SAH Specificity: ", sah_specificity)
sdh_tn, sdh_fp, sdh_fn, sdh_tp = confusion_matrix(sdh_all_labels, sdh_all_preds).ravel()
# 计算特异性
sdh_specificity = sdh_tn / (sdh_tn + sdh_fp)
print("SDH Specificity: ", sdh_specificity)

print("EDH Classification Report:")
print(classification_report(edh_all_labels, edh_all_preds, digits=4, target_names=["No Hemorrhage", "Hemorrhage"]))
print("Accuracy: ", accuracy_score(edh_all_labels, edh_all_preds))
print("IPH Classification Report:")
print(classification_report(iph_all_labels, iph_all_preds, digits=4, target_names=["No Hemorrhage", "Hemorrhage"]))
print("Accuracy: ", accuracy_score(iph_all_labels, iph_all_preds))
print("IVH Classification Report:")
print(classification_report(ivh_all_labels, ivh_all_preds, digits=4, target_names=["No Hemorrhage", "Hemorrhage"]))
print("Accuracy: ", accuracy_score(ivh_all_labels, ivh_all_preds))
print("SAH Classification Report:")
print(classification_report(sah_all_labels, sah_all_preds, digits=4, target_names=["No Hemorrhage", "Hemorrhage"]))
print("Accuracy: ", accuracy_score(sah_all_labels, sah_all_preds))
print("SDH Classification Report:")
print(classification_report(sdh_all_labels, sdh_all_preds, digits=4, target_names=["No Hemorrhage", "Hemorrhage"]))
print("Accuracy: ", accuracy_score(sdh_all_labels, sdh_all_preds))
print('宏平均指标:')
print('Macro Precision: ', precision.mean())
print('Macro Recall: ', recall.mean())
print('Macro F1: ', f1.mean())
specificity = (edh_specificity + iph_specificity + ivh_specificity + sah_specificity + sdh_specificity) / 5
print('Macro Specificity: ', specificity)
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1-score: {f1:.4f}")

with open(output_file, "w") as f:
    f.write("Sample\tTrue Label\tPredicted Label\n")
    for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
        f.write(f"{i}\t{true_label}\t{pred_label}\n")

print(f"Test results saved to {output_file}")
