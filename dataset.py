import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import json

# Carrega as classes
with open("classes.json", "r") as f:
    class_to_id = json.load(f)

# inverter: "T-Shirt" -> 17
class_to_id = {v: int(k) for k, v in class_to_id.items()}


# ==========================================
# Dataset LIMPO — sem None, sem erro de label
# ==========================================
class ClothingDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        valid_rows = []

        for _, row in df.iterrows():
            label_name = row["label"]
            img_id = row["image"]

            # pula se não existe no JSON
            if label_name not in class_to_id:
                continue

            # procura em vários formatos
            found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                if os.path.exists(os.path.join(img_dir, img_id + ext)):
                    found = True
                    break

            if found:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows)
        print(f"[OK] Imagens válidas no dataset: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row["image"]
        label_name = row["label"]
        label = class_to_id[label_name]

        # acha o arquivo real
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            test = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(test):
                img_path = test
                break

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# collate seguro
def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


def get_dataloader(csv_path, img_dir, batch_size=32, shuffle=True, img_size=448):
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # agora dinâmico
        transforms.ToTensor()
    ])

    dataset = ClothingDataset(csv_path, img_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
