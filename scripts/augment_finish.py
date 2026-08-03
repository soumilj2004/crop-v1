#!/usr/bin/env python3
"""Finish augmenting wheat to 600 per class."""
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np

WHEAT_DIR = Path("data/raw/wheat")
AUGS = ['rotate_15','rotate_neg15','rotate_30','rotate_neg30','flip_h','flip_v',
    'brightness_up','brightness_down','contrast_up','contrast_down',
    'saturation_up','saturation_down','sharpness_up','blur',
    'crop_center','crop_top_left','crop_bottom_right','shear_h','shear_v',
    'posterize','solarize','scale_up','noise','color_shift','gamma']

def do_aug(img, t, s):
    random.seed(s)
    w, h = img.size
    if t == 'rotate_15': return img.rotate(15, fillcolor=(128,128,128))
    if t == 'rotate_neg15': return img.rotate(-15, fillcolor=(128,128,128))
    if t == 'rotate_30': return img.rotate(30, fillcolor=(128,128,128))
    if t == 'rotate_neg30': return img.rotate(-30, fillcolor=(128,128,128))
    if t == 'flip_h': return img.transpose(Image.FLIP_LEFT_RIGHT)
    if t == 'flip_v': return img.transpose(Image.FLIP_TOP_BOTTOM)
    if t == 'brightness_up': return ImageEnhance.Brightness(img).enhance(1.3)
    if t == 'brightness_down': return ImageEnhance.Brightness(img).enhance(0.7)
    if t == 'contrast_up': return ImageEnhance.Contrast(img).enhance(1.4)
    if t == 'contrast_down': return ImageEnhance.Contrast(img).enhance(0.6)
    if t == 'saturation_up': return ImageEnhance.Color(img).enhance(1.5)
    if t == 'saturation_down': return ImageEnhance.Color(img).enhance(0.5)
    if t == 'sharpness_up': return ImageEnhance.Sharpness(img).enhance(2.0)
    if t == 'blur': return img.filter(ImageFilter.GaussianBlur(1.5))
    if t == 'crop_center':
        nw, nh = int(w*0.8), int(h*0.8)
        l, tp = (w-nw)//2, (h-nh)//2
        return img.crop((l, tp, l+nw, tp+nh)).resize((w,h), Image.BICUBIC)
    if t == 'crop_top_left':
        nw, nh = int(w*0.8), int(h*0.8)
        return img.crop((0, 0, nw, nh)).resize((w,h), Image.BICUBIC)
    if t == 'crop_bottom_right':
        nw, nh = int(w*0.8), int(h*0.8)
        return img.crop((w-nw, h-nh, w, h)).resize((w,h), Image.BICUBIC)
    if t == 'shear_h':
        return img.transform((w,h), Image.AFFINE, (1,0.2,0,0,1,0), fillcolor=(128,128,128))
    if t == 'shear_v':
        return img.transform((w,h), Image.AFFINE, (1,0,0,0.2,1,0), fillcolor=(128,128,128))
    if t == 'posterize': return ImageOps.posterize(img, 3)
    if t == 'solarize': return ImageOps.solarize(img, 128)
    if t == 'scale_up':
        nw, nh = int(w*1.2), int(h*1.2)
        r = img.resize((nw, nh), Image.BICUBIC)
        l, tp = (nw-w)//2, (nh-h)//2
        return r.crop((l, tp, l+w, tp+h))
    if t == 'noise':
        a = np.array(img)
        n = np.random.normal(0, 25, a.shape).astype(np.int16)
        return Image.fromarray(np.clip(a.astype(np.int16)+n, 0, 255).astype(np.uint8))
    if t == 'color_shift':
        a = np.array(img).astype(np.float32)
        sh = random.uniform(-30, 30)
        a[:,:,0] = np.clip(a[:,:,0]+sh, 0, 255)
        a[:,:,1] = np.clip(a[:,:,1]-sh/2, 0, 255)
        a[:,:,2] = np.clip(a[:,:,2]+sh/3, 0, 255)
        return Image.fromarray(a.astype(np.uint8))
    if t == 'gamma':
        g = random.uniform(0.7, 1.5)
        a = np.array(img).astype(np.float32) / 255.0
        return Image.fromarray((np.power(a, g)*255).astype(np.uint8))
    return img


TARGET = 600
for cls in ['Leaf Rust', 'Loose Smut', 'Crown & Root Rot', 'Healthy']:
    cls_dir = WHEAT_DIR / cls
    imgs = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
    cur = len(imgs)
    if cur >= TARGET:
        print(f'{cls}: {cur} (ok)')
        continue
    need = TARGET - cur
    print(f'{cls}: {cur} -> {TARGET} (+{need})')
    added = 0
    for i in range(need):
        src = random.choice(imgs)
        at = random.choice(AUGS)
        try:
            with Image.open(str(src)) as im:
                a = do_aug(im, at, random.randint(0, 100000))
                a.save(str(cls_dir / f'aug_{added:06d}.jpg'), 'JPEG', quality=95)
                added += 1
        except Exception:
            pass
    final = len(list(cls_dir.glob('*.jpg')))
    print(f'  Done: {final} total')

print('\nAll wheat classes augmented.')
