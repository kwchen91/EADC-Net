import argparse
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.datasets import SegmentationDataset, DetectionDataset
from src.models.segmentation import build_unet
from src.models.detection import build_detector


def get_device(cfg_device):
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device

def vis_seg(cfg, ckpt, limit=12):
    dev = get_device(cfg.get('device','auto'))
    data = cfg['data']; model_cfg = cfg['model']; training = cfg['training']
    ds = SegmentationDataset(data['root'], data['img_dir'], data['mask_dir'], data['mask_map'], data['img_size'], None, False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(training.get('num_workers',0)))
    model = build_unet(model_cfg['in_channels'], model_cfg['out_channels']).to(dev)
    state = torch.load(ckpt, map_location=dev); model.load_state_dict(state['state_dict']); model.eval()

    outdir = Path('runs/seg/viz'); outdir.mkdir(parents=True, exist_ok=True)
    count = 0
    with torch.no_grad():
        for img, mask in dl:
            img = img.to(dev)
            prob = torch.sigmoid(model(img))
            pred = (prob > 0.5).float().cpu().numpy()[0,0]
            raw = (img.cpu().numpy()[0,0] * 255).astype(np.uint8)
            over = Image.fromarray(raw).convert('RGB')
            gt = (mask.numpy()[0,0] > 0.5)
            rr = np.array(over)
            rr[gt, 1] = 255
            rr[pred > 0.5, 0] = 255
            Image.fromarray(rr).save(outdir / f"seg_{count:03d}.png")
            count += 1
            if count >= limit: break
    print(f"[ok] wrote {count} seg visualizations to {outdir}")

def vis_det(cfg, ckpt, score_thr=0.3, limit=20):
    dev = get_device(cfg.get('device', 'auto'))
    data = cfg['data']; model_cfg = cfg['model']
    ds = DetectionDataset(data['root'], data['img_dir'],
                          data.get('labels_csv', 'labels.csv'),
                          data.get('yolo_dir', 'labels'),
                          data['img_size'], None, False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))

    model = build_detector(model_cfg['num_classes']).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state['state_dict'])
    model.eval()

    save_dir = Path(cfg['training'].get('save_dir', 'runs/det')) / 'viz'
    save_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for images, _ in tqdm(dl, desc="[det][viz]"):
        img = images[0].to(dev)
        outputs = model([img])[0]

        boxes = outputs['boxes'].detach().cpu().numpy()
        scores = outputs['scores'].detach().cpu().numpy()

        np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        canvas = pil_img.copy()
        draw = ImageDraw.Draw(canvas)

        for box, score in zip(boxes, scores):
            if score < score_thr:
                continue
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{score:.2f}", fill="red")

        canvas.save(save_dir / f"vis_{count:03d}.png")
        count += 1
        if count >= limit:
            break

    print(f"[det][viz] saved {count} images to {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['seg','det'], required=True)
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--score_thr', type=float, default=0.3, help='det threshold')
    ap.add_argument('--limit', type=int, default=12)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.task == 'seg':
        vis_seg(cfg, args.ckpt, limit=args.limit)
    else:
        vis_det(cfg, args.ckpt, score_thr=args.score_thr, limit=args.limit)

if __name__ == '__main__':
    main()
