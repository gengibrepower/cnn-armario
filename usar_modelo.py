import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from train import ClothingCNN
import json
import sys

# --------- CONFIGS ---------
MODEL_PATH = "modelo.pth"
CLASSES_JSON = "classes.json"
IMAGE_PATH = r"c:\cnn\cnn-armario\imagens\ft_celular.jpg"  # ajuste se necessário

# --------- checar arquivos ---------
if not os.path.exists(MODEL_PATH):
    print(f"Erro: modelo não encontrado em {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(CLASSES_JSON):
    print(f"Erro: classes.json não encontrado em {CLASSES_JSON}")
    sys.exit(1)

if not os.path.exists(IMAGE_PATH):
    print(f"Erro: imagem não encontrada em {IMAGE_PATH}")
    sys.exit(1)

# --------- carregar mapeamento de classes ---------
with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    id_to_class = json.load(f)

# --------- transforms (mesmos do dataset) ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------- device (GPU se disponível) ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- carregar modelo (sem treinar) ---------
num_classes = len(id_to_class)
model = ClothingCNN(num_classes).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# --------- função de previsão (rápida) ---------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return id_to_class[str(pred)]

# --------- executar e imprimir só a label prevista ---------
start = time.time()
label = predict(IMAGE_PATH)
elapsed = (time.time() - start) * 1000  # ms
print(label)
# Se quiser ver tempo, descomente a linha abaixo:
# print(f"# tempo: {elapsed:.1f} ms")
