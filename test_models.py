import torch
from torchvision import models
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_efficientnet(model_path, num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

models_dir = Path("models")

# Crop classifier
crop_model = load_efficientnet(models_dir / "crop_efficientnet_best.pt", 2)
img = Image.open('data/split/rice/train/Healthy/100002.jpg').convert('RGB')
x = transform(img).unsqueeze(0)
with torch.no_grad():
    out = crop_model(x)
    probs = torch.softmax(out, dim=1)
    pred = torch.argmax(probs, dim=1)
crop_classes = ['rice', 'wheat']
print(f'Crop classifier: pred={crop_classes[pred.item()]}, probs={probs[0].tolist()}')

# Rice classifier
rice_model = load_efficientnet(models_dir / "rice_efficientnet_best.pt", 5)
rice_classes = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Healthy', 'Tungro']
img = Image.open('data/split/rice/train/Blast/100004.jpg').convert('RGB')
x = transform(img).unsqueeze(0)
with torch.no_grad():
    out = rice_model(x)
    probs = torch.softmax(out, dim=1)
    pred = torch.argmax(probs, dim=1)
print(f'Rice classifier: pred={rice_classes[pred.item()]}, probs={probs[0].tolist()}')

# Wheat classifier
wheat_model = load_efficientnet(models_dir / "wheat_efficientnet_best.pt", 4)
wheat_classes = ['Crown & Root Rot', 'Healthy', 'Leaf Rust', 'Loose Smut']
img = Image.open('data/split/wheat/train/Healthy/00011.jpg').convert('RGB')
x = transform(img).unsqueeze(0)
with torch.no_grad():
    out = wheat_model(x)
    probs = torch.softmax(out, dim=1)
    pred = torch.argmax(probs, dim=1)
print(f'Wheat classifier: pred={wheat_classes[pred.item()]}, probs={probs[0].tolist()}')

print("All models loaded and tested successfully!")
