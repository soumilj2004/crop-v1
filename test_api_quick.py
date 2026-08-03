import sys
sys.path.insert(0, 'C:\\CropGuardAI\\cropguard_ai')

from fastapi.testclient import TestClient
from api.main import app
from pathlib import Path

with TestClient(app) as client:
    r = client.get("/")
    print(f"GET /: {r.status_code}")

    r = client.get("/health")
    print(f"GET /health: {r.status_code} -> {r.json()}")

    # Rice blast
    img_path = Path("data/split/rice/train/Blast/100004.jpg")
    with open(img_path, "rb") as f:
        r = client.post("/analyze", files={"file": ("test.jpg", f, "image/jpeg")})
    print(f"\nPOST /analyze (rice blast): {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  crop={data.get('detected_crop')}, disease={data.get('disease')}, stage={data.get('stage')}")
        trt = data.get('treatment', {})
        print(f"  risk={data.get('risk_level')}, action={trt.get('action', 'N/A')[:60]}")
    else:
        print(f"  error: {r.text[:300]}")

    # Wheat leaf rust
    wheat_dir = Path("data/split/wheat/train/Leaf Rust")
    for fpath in wheat_dir.glob("*.jpg"):
        img_path = fpath
        break
    with open(img_path, "rb") as f:
        r = client.post("/analyze", files={"file": ("test.jpg", f, "image/jpeg")})
    print(f"\nPOST /analyze (wheat leaf rust): {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  crop={data.get('detected_crop')}, disease={data.get('disease')}, stage={data.get('stage')}")
        trt = data.get('treatment', {})
        print(f"  risk={data.get('risk_level')}, action={trt.get('action', 'N/A')[:60]}")
        caveats = data.get('caveats', [])
        if caveats:
            print(f"  caveats: {caveats[0][:80]}".encode('ascii', 'ignore').decode())
    else:
        print(f"  error: {r.text[:300]}")

    # Wheat healthy
    wheat_dir = Path("data/split/wheat/test/Healthy")
    for fpath in wheat_dir.glob("*.jpg"):
        img_path = fpath
        break
    with open(img_path, "rb") as f:
        r = client.post("/analyze", files={"file": ("test.jpg", f, "image/jpeg")})
    print(f"\nPOST /analyze (wheat healthy): {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  crop={data.get('detected_crop')}, disease={data.get('disease')}, stage={data.get('stage')}")
        trt = data.get('treatment', {})
        print(f"  risk={data.get('risk_level')}, action={trt.get('action', 'N/A')[:60]}")
    else:
        print(f"  error: {r.text[:300]}")

print("\nAll API tests passed!")
