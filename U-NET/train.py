import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import OxfordPetSegDataset
from model.unet import UNET
from config import BATCH_SIZE, LR, EPOCHS, DEVICE
import matplotlib.pyplot as plt

# Dataloader
train_loader = DataLoader(OxfordPetSegDataset('./data', train=True, augment=True), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(OxfordPetSegDataset('./data', train=False), batch_size=BATCH_SIZE)

# Model, loss, optimizer
model =  UNET().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def dice_score(preds, targets, num_classes=3, smooth=1e-6):
    preds = torch.argmax(preds, dim=1)
    dice_total = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_total += dice
    return dice_total / num_classes

best_val_dice = 0.0

history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_dice': [],
    'val_dice': []
}

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct_pixels = 0
    total_pixels = 0
    dice_total_train = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()
        dice_total_train += dice_score(outputs, masks)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct_pixels / total_pixels
    train_dice = dice_total_train / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    dice_total_val = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == masks).sum().item()
            val_total += masks.numel()
            dice_total_val += dice_score(outputs, masks)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_dice = dice_total_val / len(val_loader)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "model.pth")
        print(f" Saved best model at epoch {epoch+1} with val_dice={val_dice:.4f}")

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} - Acc: {train_acc:.4f} - Dice: {train_dice:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} - Acc: {val_acc:.4f} - Dice: {val_dice:.4f}")
    history['train_loss'].append(float(avg_train_loss))
    history['val_loss'].append(float(avg_val_loss))
    history['train_acc'].append(float(train_acc))
    history['val_acc'].append(float(val_acc))
    history['train_dice'].append(float(train_dice))
    history['val_dice'].append(float(val_dice))


plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 3, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Dice
plt.subplot(1, 3, 3)
plt.plot(history['train_dice'], label='Train Dice')
plt.plot(history['val_dice'], label='Val Dice')
plt.title('Dice Score')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.tight_layout()
plt.show()
