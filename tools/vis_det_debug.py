from pathlib import Path
import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from src.data.datasets import DetectionDataset
from src.models.detection import build_detector

def det_collate(b): return tuple(zip(*b))

def draw(rgb, boxes, color):
    im = rgb.copy()
    dr = ImageDraw.Draw(im)
    for x1,y1,x2,y2 in boxes:
        dr.rectangle([float(x1),float(y1),float(x2),float(y2)], outline=color, width=2)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num', type=int, default=8)
    ap.add_argument('--thr', type=float, default=0.05)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = DetectionDataset(cfg['data']['root'], cfg['data']['img_dir'],
                          cfg['data'].get('labels_csv','labels.csv'),
                          cfg['data'].get('yolo_dir','labels'),
                          cfg['data']['img_size'], None, False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=det_collate, num_workers=0)
    model = build_detector(cfg['model']['num_classes']).to(dev).eval()
    state = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(state['state_dict'])
    outdir = Path("runs/det/vis"); outdir.mkdir(parents=True, exist_ok=True)
    cnt=0
    with torch.no_grad():
        for images, targets in dl:
            img = images[0].to(dev)
            out = model([img])[0]
            keep = out['scores'].cpu().numpy() >= args.thr
            pd = out['boxes'].cpu().numpy()[keep]
            base = (images[0].cpu().numpy()[0]*255).astype(np.uint8)
            base = np.stack([base]*3, axis=-1)
            L = draw(Image.fromarray(base), targets[0]['boxes'].cpu().numpy(), (0,255,0))
            R = draw(Image.fromarray(base), pd, (255,0,0))
            canvas = Image.new('RGB', (L.width+R.width+10, max(L.height,R.height)), (30,30,30))
            canvas.paste(L, (0,0)); canvas.paste(R, (L.width+10,0))
            canvas.save(outdir / f"vis_{cnt:03d}.png"); cnt+=1
            if cnt>=args.num: break
    print(f"Saved {cnt} images to {outdir}")

if __name__ == "__main__":
    main()
