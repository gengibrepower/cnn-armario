import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torchvision import transforms 
import json
import numpy as np

# --- VARIAVEIS DE CONFIGURACAO (Constantes) ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

GLOBAL_CLASS_TO_ID = None

# --- FUNCAO DE CARREGAMENTO SEGURO ---
def _load_class_mapping():
    """Carrega o mapeamento de classes de forma segura para multiprocessing."""
    global GLOBAL_CLASS_TO_ID
    
    # Se ja foi carregado, apenas retorna o mapeamento
    if GLOBAL_CLASS_TO_ID is not None:
        return GLOBAL_CLASS_TO_ID

    try:
        with open(CLASSES_PATH, "r") as f:
            class_to_id = json.load(f)
        
        # Inverter e garantir que a chave e int (se o JSON usar strings para IDs)
        GLOBAL_CLASS_TO_ID = {v: int(k) for k, v in class_to_id.items()}
        return GLOBAL_CLASS_TO_ID
    
    except FileNotFoundError:
        print(f"Erro: Arquivo de classes nao encontrado em {CLASSES_PATH}")
        # Retornar um dicionario vazio ou levantar erro, dependendo do seu requisito
        return {}


class ClothingDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        
        # Carrega o mapeamento aqui
        CLASS_TO_ID = _load_class_mapping()
        if not CLASS_TO_ID:
            raise RuntimeError("Nao foi possivel carregar o mapeamento de classes.")
            
        self.img_dir = img_dir
        self.transform = transform
        self._class_to_id = CLASS_TO_ID # Armazena o mapeamento na instancia

        df = pd.read_csv(csv_path)
        valid_rows = []

        for _, row in df.iterrows():
            label_name = row["label"]
            img_id = row["image"]

            if label_name not in self._class_to_id:
                continue

            found = False
            for ext in [".jpg", ".jpeg", ".png"]:
                test_path = os.path.join(img_dir, img_id + ext)
                if os.path.exists(test_path):
                    found = True
                    row["_img_path"] = test_path 
                    break

            if found:
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        
        # otimizando armazena todas as labels como um tensor na memoria
        label_names = self.df["label"].map(self._class_to_id).tolist()
        self.labels = torch.tensor(label_names, dtype=torch.long)
        
        print(f"[OK] Imagens validas no dataset: {len(self.df)}")


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["_img_path"]
        label_name = row["label"]
        # Usa o mapeamento armazenado na instancia
        label = self._class_to_id[label_name] 

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_all_labels(self):
        return self.labels


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)


def get_dataloader(csv_path, img_dir, batch_size=32, img_size=224, transforms=None, split='train', num_workers=0, pin_memory=False):
    
    full_dataset = ClothingDataset(csv_path, img_dir, transform=transforms)
    
    data_len = len(full_dataset)
    
    train_size = int(TRAIN_RATIO * data_len)
    val_size = int(VAL_RATIO * data_len)
    test_size = data_len - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(data_len, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    if split == 'train':
        subset = Subset(full_dataset, train_indices)
        shuffle_data = True
    elif split == 'val':
        subset = Subset(full_dataset, val_indices)
        shuffle_data = False
    elif split == 'test':
        subset = Subset(full_dataset, test_indices)
        shuffle_data = False
    else:
        raise ValueError(f"Split '{split}' nao reconhecido. Use 'train', 'val', ou 'test'.")

    print(f"[{split.upper()}] Tamanho do dataset: {len(subset)}")
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )