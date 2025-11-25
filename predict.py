import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import logging
import os
from train import ClothingCNNComplex 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")


ID_TO_CLASS = {}
try:
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        ID_TO_CLASS = {k: v for k, v in json.load(f).items()} 
    logging.info(f"Classes carregadas: {len(ID_TO_CLASS)} categorias.")
except FileNotFoundError:
    logging.error(f"O arquivo de classes '{CLASSES_PATH}' não foi encontrado.")
except Exception as e:
    logging.error(f"Erro ao carregar classes.json: {e}")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Dispositivo de inferência: {DEVICE}")


MODEL = None
NUM_CLASSES = len(ID_TO_CLASS)

try:

    MODEL = ClothingCNNComplex(NUM_CLASSES).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True) 
    
    MODEL.load_state_dict(state, strict=False)
    MODEL.eval()
    logging.info(f"Modelo PyTorch '{os.path.basename(MODEL_PATH)}' carregado com sucesso.")
except FileNotFoundError:
    logging.error(f"O arquivo do modelo '{os.path.basename(MODEL_PATH)}' não foi encontrado. O ML está desabilitado.")
except Exception as e:
    logging.error(f"Erro CRÍTICO ao carregar o modelo PyTorch: {e}. O ML está desabilitado.")
    MODEL = None



@torch.inference_mode()
def predict_category(filepath):
    if MODEL is None:
        return "Desconhecida (Modelo indisponível)"
        
    try:
        img = Image.open(filepath).convert("RGB")

        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        out = MODEL(x)

        idx = torch.argmax(out, 1).item()

        return ID_TO_CLASS.get(str(idx), "Desconhecida (Novo Índice)")
        
    except Exception as e:
        logging.error(f"Erro na inferência para {filepath}: {e}")
        return "Desconhecida (Erro na Inferência)"
    #isso msm