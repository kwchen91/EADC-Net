
import os
import argparse
from PIL import Image
import numpy as np

def overlay_mask(image_path, mask_path, out_path, alpha=0.35):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  
    img_np = np.array(img).astype(np.float32)

    mask_np = np.array(mask)
    if mask_np.max() <= 1:
        mask_np = (mask_np > 0).astype(np.uint8) * 255
    else:
        mask_np = (mask_np > 127).astype(np.uint8) * 255

    color = np.zeros_like(img_np)
    color[..., 1] = mask_np 

    out = (img_np * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)
    print(f"Saved overlay to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="path to input image")
    parser.add_argument("--mask", required=True, help="path to mask (GT or predicted)")
    parser.add_argument("--out", required=True, help="path to save overlay png")
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    overlay_mask(args.img, args.mask, args.out, alpha=args.alpha)
