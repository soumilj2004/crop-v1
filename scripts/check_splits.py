from pathlib import Path
BASE = Path('C:/CropGuardAI/cropguard_ai/data/split')
for crop in ['rice','wheat']:
    for split in ['train','val','test']:
        p = BASE / crop / split
        if p.exists():
            n = sum(1 for _ in p.rglob('*') if _.is_file())
            print(f'{crop}/{split}: {n}')
        else:
            print(f'{crop}/{split}: MISSING')