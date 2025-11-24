import os
import time
import torch
from PIL import Image
import torchvision.transforms as transforms
from train import ClothingCNNComplex
import json
import sys

MODEL_PATH = "modelo_complexo.pth"
CLASSES_JSON = "classes.json"
IMAGE_PATH = r"c:\cnn\cnn-armario\imagens\ft_celular.jpg"

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_JSON) or not os.path.exists(IMAGE_PATH):
    print("Erro: arquivo necessário não encontrado.")
    sys.exit(1)

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    id_to_class = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(id_to_class)
model = ClothingCNNComplex(num_classes).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

torch.backends.cudnn.benchmark = True

@torch.inference_mode()
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    out = model(x)
    return id_to_class[str(torch.argmax(out, dim=1).item())]

start = time.time()
label = predict(IMAGE_PATH)
print(label)
