import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# ============================
# CONFIG
# ============================
CSV_PATH = r"C:\cnn\clothing-dataset\images.csv"
IMAGES_PATH = r"C:\cnn\clothing-dataset\images_original"

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)


# ============================
# DATASET
# ============================
class ClothingDataset(Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        self.data = pd.read_csv(csv_path)
        labels = sorted(self.data["label"].unique())
        self.label_to_id = {lab: i for i, lab in enumerate(labels)}
        self.id_to_label = {i: lab for lab, i in self.label_to_id.items()}
        self.images_path = images_path
        self.transform = transform

        # Remover linhas sem label
        self.data = self.data[self.data["label"].notna()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_id = row["image"]
        img_path = os.path.join(self.images_path, img_id + ".jpg")

        image = Image.open(img_path).convert("RGB")

        label_str = row["label"]
        label = self.label_to_id[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================
# TRANSFORMAÇÕES
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================
# MODELO CNN
# ============================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(SimpleCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================
# MAIN
# ============================
from tqdm import tqdm

# ============================
# TREINO COM PROGRESS BAR
# ============================
if __name__ == "__main__":
    dataset = ClothingDataset(CSV_PATH, IMAGES_PATH, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print("Total de imagens:", len(dataset))

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        running_loss = 0

        print(f"\nTreinando época {epoch+1}/{EPOCHS}")
        for images, labels in tqdm(loader, desc=f"Época {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Loss da época {epoch+1}: {running_loss:.4f}")

    print("Treino concluído!")
