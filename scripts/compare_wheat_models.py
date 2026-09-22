import torch, time
from pathlib import Path
from torchvision import models, transforms
from PIL import Image

t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
CLASSES = ['Crown & Root Rot', 'Healthy', 'Leaf Rust', 'Loose Smut']

def load(p):
    m = models.efficientnet_b0(weights=None)
    m.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3), torch.nn.Linear(1280, 512),
        torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(512, 4))
    m.load_state_dict(torch.load(p, map_location='cpu'))
    m.eval()
    return m

def eval_model(m, label):
    correct = total = 0
    t0 = time.time()
    with torch.no_grad():
        for i, cls in enumerate(CLASSES):
            d = Path('data/split/wheat/test') / cls
            for f in d.iterdir():
                if not f.is_file():
                    continue
                try:
                    img = Image.open(f).convert('RGB')
                except Exception:
                    continue
                x = t(img).unsqueeze(0)
                correct += (m(x).argmax(1).item() == i)
                total += 1
    print(f'{label}: {correct}/{total} = {correct/total:.4f} ({time.time()-t0:.0f}s)', flush=True)

if __name__ == '__main__':
    print('start', flush=True)
    eval_model(load(r'C:\Users\Lenovo\AppData\Local\Temp\opencode\wheat_friend\rebuilt_archive.pt'), 'FRIEND (GPU)  ')
    eval_model(load('models/wheat_efficientnet_best.pt'), 'OURS (local)   ')
    print('done', flush=True)
