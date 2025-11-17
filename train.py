import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
import json
from tqdm import tqdm
import torch.nn.functional as F

# ======================
# CONFIGURAÇÕES
# ======================
CSV_PATH = r"C:\cnn\clothing-dataset\images.csv"
IMG_DIR = r"C:\cnn\clothing-dataset\images_original"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 0.0005

# carregar classes
with open("classes.json", "r") as f:
    class_map = json.load(f)

NUM_CLASSES = len(class_map)


# ======================
# MODELO CNN SIMPLES
# ======================
class ClothingCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# ======================
# FUNÇÃO DE TREINO
# ======================
def treinar():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Usando device:", device)

    loader = get_dataloader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE)

    model = ClothingCNN(NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Loss médio: {total_loss / len(loader):.4f}")

    # ======================
    # SALVA O MODELO
    # ======================
    torch.save(model.state_dict(), "modelo.pth")
    print("\n[OK] Modelo salvo como modelo.pth")


# ======================
# BLOCO PRINCIPAL
# ======================
if __name__ == "__main__":
    treinar()
