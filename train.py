import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from dataset import get_dataloader  
import json
from tqdm import tqdm
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau 

CSV_PATH = r"C:\cnn\clothing-dataset\images.csv"
IMG_DIR = r"C:\cnn\clothing-dataset\images_original"

NUM_EPOCHS = 30 
BATCH_SIZE = 32
LR = 1e-3
IMG_SIZE = 224
PATIENCE = 5 

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

try:
    with open("classes.json", "r") as f:
        class_map = json.load(f)
    NUM_CLASSES = len(class_map)
except FileNotFoundError:
    NUM_CLASSES = 10


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size, is_train=True):
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_pretrained_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    

    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    
    return model


ClothingCNNComplex = get_pretrained_model 


def treinar():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = get_dataloader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, 
                                  transforms=get_transforms(IMG_SIZE, is_train=True), split='train', 
                                  num_workers=4, pin_memory=(device == 'cuda'))

    print("Coletando labels para cálculo de pesos (Otimizado)...")
    train_subset = train_loader.dataset 

    full_labels = train_subset.dataset.get_all_labels() 
    all_labels_tensor = full_labels[train_subset.indices] 

    all_labels_numpy = all_labels_tensor.cpu().numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels_numpy), y=all_labels_numpy)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Cálculo de pesos concluído.")
    
    val_loader = get_dataloader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, 
                                transforms=get_transforms(IMG_SIZE, is_train=False), split='val', 
                                num_workers=4, pin_memory=(device == 'cuda'))
    test_loader = get_dataloader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, 
                                 transforms=get_transforms(IMG_SIZE, is_train=False), split='test', 
                                 num_workers=4, pin_memory=(device == 'cuda'))

    model = get_pretrained_model(NUM_CLASSES).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.layer4.parameters(), 'lr': LR / 10},  
        {'params': model.fc.parameters(), 'lr': LR}           
    ], weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler()
    
    accuracy = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)
    f1 = F1Score(task='multiclass', num_classes=NUM_CLASSES, average='macro').to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                accuracy.update(preds, labels)
                f1.update(preds, labels)
        
        val_loss /= len(val_loader)
        val_acc = accuracy.compute()
        val_f1 = f1.compute()
        accuracy.reset()
        f1.reset()
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test Evaluation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            test_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            accuracy.update(preds, labels)
            f1.update(preds, labels)
    
    test_loss /= len(test_loader)
    test_acc = accuracy.compute()
    test_f1 = f1.compute()
    print(f"Final Test Results -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1-Score: {test_f1:.4f}")

if __name__ == "__main__":
    treinar()