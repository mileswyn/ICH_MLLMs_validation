import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataset.dataset import HemorrhageDataset, get_transforms
from model.resnet50 import get_resnet50
from model.vit import get_vit_b, get_vit_l
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "resnet50"
model_path = "./best_model_resnet50.pth"
data_dir = "./data"
batch_size = 1
output_file = "./test_results_resnet50.txt"

test_dataset = HemorrhageDataset(data_dir=data_dir, mode='test', transform=get_transforms('test'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if model_name == "resnet50":
    model = get_resnet50(num_classes=2).to(device)
elif model_name == "vit":
    model = get_vit_b(num_classes=2).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print("Classification Report:")
print(classification_report(all_labels, all_preds, digits=4, target_names=["No Hemorrhage", "Hemorrhage"]))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

with open(output_file, "w") as f:
    f.write("Sample\tTrue Label\tPredicted Label\n")
    for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
        f.write(f"{i}\t{true_label}\t{pred_label}\n")

print(f"Test results saved to {output_file}")
