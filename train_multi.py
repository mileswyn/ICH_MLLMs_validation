# -*- encoding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataset.dataset_multi import HemorrhageDataset, get_transforms
from model.resnet50 import get_resnet50
import wandb
from tqdm import tqdm

wandb.init(project="hemorrhage-classification-multi-resnet50", config={
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-4,
    "model": "resnet50", 
})

# 假设有 5 个类别，每个类别的正样本数如下
positive_counts = torch.tensor([164, 798, 650, 878, 683])  # 每个类别的正样本数
negative_counts = torch.tensor([4890, 4256, 4404, 4176, 4371])  # 每个类别的负样本数

pos_weight = negative_counts / positive_counts
print(f"Pos weight: {pos_weight}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = wandb.config.batch_size
epochs = wandb.config.epochs
learning_rate = wandb.config.learning_rate
model_name = wandb.config.model

train_dataset = HemorrhageDataset(data_dir='./data', mode='train', transform=get_transforms('train'))
val_dataset = HemorrhageDataset(data_dir='./data', mode='val', transform=get_transforms('val'))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

if model_name == "resnet50":
    model = get_resnet50(num_classes=5).to(device)
elif model_name == "vit":
    model = get_vit(num_classes=5).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
save_path = "./best_model_multi_resnet50.pth"
early_stop = 0
best_val_f1 = 0.0

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    edh_correct, iph_correct, ivh_correct, sah_correct, sdh_correct = 0, 0, 0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.sigmoid(outputs) > 0.5
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        edh_correct += predicted[:, 0].eq(labels[:, 0]).sum().item()
        iph_correct += predicted[:, 1].eq(labels[:, 1]).sum().item()
        ivh_correct += predicted[:, 2].eq(labels[:, 2]).sum().item()
        sah_correct += predicted[:, 3].eq(labels[:, 3]).sum().item()
        sdh_correct += predicted[:, 4].eq(labels[:, 4]).sum().item()

    train_acc = 100. * train_correct / train_total
    edh_acc = 100. * edh_correct / train_total
    iph_acc = 100. * iph_correct / train_total
    ivh_acc = 100. * ivh_correct / train_total
    sah_acc = 100. * sah_correct / train_total
    sdh_acc = 100. * sdh_correct / train_total
    wandb.log({"train_loss": train_loss / len(train_loader), "train_acc": train_acc,
               "edh_acc": edh_acc, "iph_acc": iph_acc, "ivh_acc": ivh_acc, 
               "sah_acc": sah_acc, "sdh_acc": sdh_acc})

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_edh_correct, val_iph_correct, val_ivh_correct, val_sah_correct, val_sdh_correct = 0, 0, 0, 0, 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = torch.sigmoid(outputs) > 0.5
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            val_edh_correct += predicted[:, 0].eq(labels[:, 0]).sum().item()
            val_iph_correct += predicted[:, 1].eq(labels[:, 1]).sum().item()
            val_ivh_correct += predicted[:, 2].eq(labels[:, 2]).sum().item()
            val_sah_correct += predicted[:, 3].eq(labels[:, 3]).sum().item()
            val_sdh_correct += predicted[:, 4].eq(labels[:, 4]).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = 100. * val_correct / val_total
    val_edh_acc = 100. * val_edh_correct / val_total
    val_iph_acc = 100. * val_iph_correct / val_total
    val_ivh_acc = 100. * val_ivh_correct / val_total
    val_sah_acc = 100. * val_sah_correct / val_total
    val_sdh_acc = 100. * val_sdh_correct / val_total
    val_precision = precision_score(all_labels, all_preds, average=None)
    val_recall = recall_score(all_labels, all_preds, average=None)
    val_f1 = f1_score(all_labels, all_preds, average=None)

    wandb.log({"val_loss": val_loss / len(val_loader), "val_acc": val_acc, "val_prec": val_precision, "val_recall": val_recall, "val_f1": val_f1,
               "val_edh_acc": val_edh_acc, "val_iph_acc": val_iph_acc, 
               "val_ivh_acc": val_ivh_acc, "val_sah_acc": val_sah_acc, 
               "val_sdh_acc": val_sdh_acc})

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Train EDH Acc: {edh_acc:.2f}%, Train IPH Acc: {iph_acc:.2f}%, Train IVH Acc: {ivh_acc:.2f}%, Train SAH Acc: {sah_acc:.2f}%, Train SDH Acc: {sdh_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
          f"Val EDH Acc: {val_edh_acc:.2f}%, Val IPH Acc: {val_iph_acc:.2f}%, Val IVH Acc: {val_ivh_acc:.2f}%, Val SAH Acc: {val_sah_acc:.2f}%, Val SDH Acc: {val_sdh_acc:.2f}%, ")
    print("val_f1: ", val_f1)
    print("val_precision: ", val_precision)
    print("val_recall: ", val_recall)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_f1 = val_f1.sum() / 5
    # if avg_val_loss < best_val_loss:
    if avg_val_f1 > best_val_f1:
        best_val_loss = avg_val_loss
        best_val_f1 = avg_val_f1
        early_stop = 0
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f} Val Acc {val_acc:.4f},"
              f"Val EDH Acc: {val_edh_acc:.2f}%, Val IPH Acc: {val_iph_acc:.2f}%, Val IVH Acc: {val_ivh_acc:.2f}%, Val SAH Acc: {val_sah_acc:.2f}%, Val SDH Acc: {val_sdh_acc:.2f}%")
        print("Best val_f1: ", val_f1)
        print("Best val_precision: ", val_precision)
        print("Best val_recall: ", val_recall)
    else:
        early_stop += 1
        if early_stop > 30:
            print("Early stopping")
            break

print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
