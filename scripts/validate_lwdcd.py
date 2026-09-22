import os
import sys
from PIL import Image

root = r"C:\CropGuardAI\datasets\lwdcd2020\Large Wheat Disease Classification Dataset"
bad = []
count = 0
for cls in os.listdir(root):
    cdir = os.path.join(root, cls)
    if not os.path.isdir(cdir):
        continue
    for f in os.listdir(cdir):
        p = os.path.join(cdir, f)
        count += 1
        try:
            with Image.open(p) as im:
                im.load()
        except Exception as e:
            bad.append((p, str(e)))
print(f"checked {count}, bad {len(bad)}")
for p, e in bad:
    print("BAD:", p, "->", e)
