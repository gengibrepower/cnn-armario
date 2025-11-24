# Conteúdo do predict.py (Versão Refatorada)
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
import os
from train import ClothingCNNComplex 

# ... (Configuração de Logging) ...

# --- Configuração Global ---

# Carregamento seguro e com fallback
try:
    with open("classes.json", "r", encoding="utf-8") as f:
        ID_TO_CLASS = json.load(f)
except FileNotFoundError:
    logging.error("O arquivo 'classes.json' não foi encontrado.")
    ID_TO_CLASS = {}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuração do Modelo (Com tratamento de erro) ---
MODEL = None
try:
    # Cria o modelo (a classe deve ser importada de train.py)
    MODEL = ClothingCNNComplex(len(ID_TO_CLASS)).to(DEVICE)
    
    # SEGURANÇA APLICADA AQUI: weights_only=True
    state = torch.load("modelo.pth", map_location=DEVICE, weights_only=True) 
    MODEL.load_state_dict(state, strict=False)
    MODEL.eval()
    logging.info(f"Modelo PyTorch carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo PyTorch: {e}. A API continuará sem predição.")

# --- Função de Predição (Com tratamento de erro) ---

@torch.inference_mode()
def predict_category(filepath):
    """
    Executa a inferência de ML e retorna a categoria.
    """
    if not MODEL:
        return "Desconhecida (Modelo indisponível)"
        
    try:
        img = Image.open(filepath).convert("RGB")
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        out = MODEL(x)
        idx = torch.argmax(out, 1).item()
        
        # Robustez: Usa .get() para retornar 'Desconhecida' se a chave não existir
        return ID_TO_CLASS.get(str(idx), "Desconhecida")
        
    except Exception as e:
        logging.error(f"Erro na inferência para {filepath}: {e}")
        return "Desconhecida (Erro na Inferência)"