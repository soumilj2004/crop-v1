from pathlib import Path
SPLIT = Path('C:/CropGuardAI/cropguard_ai/data/split')
for s in ['train','val','test']:
    n = sum(1 for _ in (SPLIT/'crop'/s).rglob('*') if _.is_file())
    print(f'crop/{s}: {n}')