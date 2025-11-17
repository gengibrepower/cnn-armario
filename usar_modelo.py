import torch
from PIL import Image
import torchvision.transforms as transforms
from train import ClothingCNN
import json

MODEL_PATH = "modelo.pth"

# carregar classes
with open("classes.json", "r") as f:
    class_map = json.load(f)

# TRANSFORM CORRETO — O MESMO DO MODELO
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # <<< CORRIGIDO
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # <<< 3 canais (RGB)
])

# carregar modelo
def carregar_modelo():
    num_classes = len(class_map)
    model = ClothingCNN(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = carregar_modelo()

# prever
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return class_map[str(predicted.item())]

# exemplo
foto = r"c:\cnn\cnn-armario\imagens\ft_celular.jpg"
print("Classe prevista:", predict(foto))
