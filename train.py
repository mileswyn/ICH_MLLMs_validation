# -*- encoding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from dataset.dataset import HemorrhageDataset, get_transforms
from model.resnet50 import get_resnet50
from model.vit import get_vit
import wandb
from tqdm import tqdm

wandb.init(project="hemorrhage-classification", config={
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-4,
    "model": "resnet50", 
})

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
    model = get_resnet50(num_classes=2).to(device)
elif model_name == "vit":
    model = get_vit(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
save_path = "./best_model.pth"
early_stop = 0

for epoch in tqdm(range(epochs)):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    train_acc = 100. * train_correct / train_total
    wandb.log({"train_loss": train_loss / len(train_loader), "train_acc": train_acc})

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = 100. * val_correct / val_total
    val_f1 = f1_score(all_labels, all_preds, average='weighted')

    wandb.log({"val_loss": val_loss / len(val_loader), "val_acc": val_acc, "val_f1": val_f1})

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")

    avg_val_loss = val_loss / len(val_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop = 0
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f} and Val Acc {val_acc:.4f} and Val_F1 {val_f1:.4f}")
    else:
        early_stop += 1
        if early_stop > 10:
            print("Early stopping")
            break

print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
