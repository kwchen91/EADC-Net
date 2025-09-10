import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def set_seed(s=42):
    random.seed(s); np.random.seed(s)

def synth_seg_pair(H=512, W=512):
    img = Image.fromarray((np.random.normal(128, 18, (H,W)).clip(0,255)).astype(np.uint8))
    m = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(m)
    y = H//5
    for _ in range(random.randint(4,5)):
        h = random.randint(60, 85)
        w = random.randint(220, 280)
        x = W//2 - w//2 + random.randint(-25,25)
        draw.rectangle([x, y, x+w, y+h], fill=255)
        y += h + random.randint(10, 30)
    img = Image.fromarray((np.array(img) + (np.array(m)//8)).clip(0,255).astype(np.uint8))
    return img, m

def bb_from_mask(mask_np):
    ys, xs = np.where(mask_np>0)
    if len(xs)==0: return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def synth_det_from_seg(mask_pil):
    m = (np.array(mask_pil)>0).astype(np.uint8)
    H, W = m.shape
    bands = np.array_split(np.arange(H), random.randint(4,5))
    boxes=[]
    for band in bands:
        sub = np.zeros_like(m); sub[band.min():band.max()+1,:]=m[band.min():band.max()+1,:]
        bb = bb_from_mask(sub)
        if bb: boxes.append(bb)
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    root = Path(args.out)
    seg_img = root/"seg"/"images"; seg_msk = root/"seg"/"masks"
    det_img = root/"det"/"images"; det_lbl = root/"det"/"labels"
    for p in [seg_img, seg_msk, det_img, det_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    csv_lines = ["image,xmin,ymin,xmax,ymax,class"]
    for i in range(args.num):
        fn = f"img_{i:04d}.png"
        img, mask = synth_seg_pair(args.img_size, args.img_size)
        img.save(seg_img/fn); mask.save(seg_msk/fn)
        img.save(det_img/fn)
        boxes = synth_det_from_seg(mask)
        h, w = mask.size[1], mask.size[0]
        yolo_lines=[]
        for (x1,y1,x2,y2) in boxes:
            csv_lines.append(f"{fn},{x1},{y1},{x2},{y2},vertebra")
            cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h; bw=(x2-x1)/w; bh=(y2-y1)/h
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (det_lbl/f"{Path(fn).stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    (root/"det"/"labels.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    (root/"det"/"classes.txt").write_text("vertebra\n", encoding="utf-8")
    print(f"[done] synthetic data written to: {root.resolve()}")

if __name__ == "__main__":
    main()