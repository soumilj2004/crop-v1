import sys
sys.path.insert(0, 'C:/CropGuardAI/cropguard_ai/scripts')
from train_all import CropDataset, get_transforms
ds = CropDataset('train', get_transforms('val'))
print(f'Train samples: {len(ds)}')
ds2 = CropDataset('val', get_transforms('val'))
print(f'Val samples: {len(ds2)}')