import json
from collections import Counter
from pathlib import Path

for crop in ['wheat', 'rice']:
    p = Path('C:/CropGuardAI/cropguard_ai/data/stage_labels') / f'{crop}_hsv_stage_labels.json'
    manifest = json.load(open(p))
    stages = Counter(manifest.values())
    print(f'{crop}: {len(manifest)} total')
    for s in ['early', 'mid', 'late']:
        print(f'  {s}: {stages.get(s, 0)}')

    diseases = {
        'wheat': ['Crown & Root Rot', 'Healthy', 'Leaf Rust', 'Loose Smut'],
        'rice': ['Bacterial Blight', 'Blast', 'Brown Spot', 'Healthy', 'Tungro'],
    }
    from collections import defaultdict
    per_disease = defaultdict(lambda: Counter())
    for rel, stage in manifest.items():
        parts = Path(rel).parts
        disease = parts[1] if len(parts) >= 3 else '?'
        per_disease[disease][stage] += 1
    for d in diseases[crop]:
        dc = per_disease.get(d, Counter())
        total = sum(dc.values())
        if total:
            e, m, l = dc.get('early', 0), dc.get('mid', 0), dc.get('late', 0)
            print(f'  {d}: {total} -> early={e} mid={m} late={l}')
    print()
