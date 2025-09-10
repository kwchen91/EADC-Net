import argparse, yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.datasets import SegmentationDataset, DetectionDataset
from src.models.segmentation import build_unet
from src.models.detection import build_detector

def get_device(cfg_device):
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device

def det_collate(batch):
    return tuple(zip(*batch))

def _iou_xyxy(a, b):
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    ixmin = np.maximum(a[:, None, 0], b[None, :, 0])
    iymin = np.maximum(a[:, None, 1], b[None, :, 1])
    ixmax = np.minimum(a[:, None, 2], b[None, :, 2])
    iymax = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ixmax - ixmin, 0, None)
    ih = np.clip(iymax - iymin, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)

def _ap_from_pr(prec, rec):
    if len(prec) == 0: return 0.0
    order = np.argsort(rec)
    rec = np.array(rec)[order]
    prec = np.array(prec)[order]
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i] - rec[i-1]) * max(prec[i], prec[i-1])
    return float(ap)

def evaluate_map_at_05(all_preds, all_gts):
    scores, matches = [], []
    total_gts = 0
    for preds, gts in zip(all_preds, all_gts):
        pboxes = np.asarray(preds.get("boxes", []), dtype=np.float32)
        pscores = np.asarray(preds.get("scores", []), dtype=np.float32)
        gboxes = np.asarray(gts.get("boxes", []), dtype=np.float32)
        total_gts += len(gboxes)
        used = np.zeros(len(gboxes), dtype=bool)
        iou = _iou_xyxy(pboxes, gboxes) if len(pboxes) and len(gboxes) else np.zeros((len(pboxes), 0), np.float32)
        for i in range(len(pboxes)):
            scores.append(float(pscores[i] if i < len(pscores) else 0.0))
            j = int(iou[i].argmax()) if iou.shape[1] > 0 else -1
            ok = (j >= 0) and (iou[i, j] >= 0.5) and (not used[j])
            matches.append(1 if ok else 0)
            if ok: used[j] = True
    order = np.argsort(scores)[::-1]
    tp = 0; fp = 0; prec = []; rec = []
    for k in order:
        if matches[k] == 1: tp += 1
        else: fp += 1
        prec.append(tp / max(tp + fp, 1))
        rec.append(tp / max(total_gts, 1))
    ap = _ap_from_pr(prec, rec)
    mAP = ap
    P = prec[-1] if prec else 0.0
    R = rec[-1] if rec else 0.0
    return mAP, P, R

def eval_seg(cfg, ckpt):
    dev = get_device(cfg.get('device','auto'))
    data = cfg['data']; model_cfg = cfg['model']
    ds = SegmentationDataset(data['root'], data['img_dir'], data['mask_dir'], data['mask_map'],
                             data['img_size'], None, augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(cfg['training'].get('num_workers',0)))
    model = build_unet(model_cfg['in_channels'], model_cfg['out_channels']).to(dev)
    try:
        state = torch.load(ckpt, map_location=dev, weights_only=True)['state_dict']  
    except TypeError:
        state = torch.load(ckpt, map_location=dev)['state_dict']
    model.load_state_dict(state); model.eval()
    dices = []
    with torch.no_grad():
        for img, mask in tqdm(dl, desc="[seg][eval]"):
            img, mask = img.to(dev), mask.to(dev)
            prob = torch.sigmoid(model(img))
            pred = (prob > 0.5).float()
            inter = (pred * mask).sum().item()
            den = pred.sum().item() + mask.sum().item() + 1e-6
            dices.append(2 * inter / den)
    md = float(np.mean(dices)) if dices else 0.0
    print(f"[seg] Dice={md:.4f}")

def eval_det(cfg, ckpt, score_thr=0.05):
    dev = get_device(cfg.get('device','auto'))
    data = cfg['data']; model_cfg = cfg['model']; training = cfg['training']
    ds = DetectionDataset(data['root'], data['img_dir'],
                          data.get('labels_csv','labels.csv'),
                          data.get('yolo_dir','labels'),
                          data['img_size'], None, False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: tuple(zip(*b)),
                    num_workers=int(training.get('num_workers',0)))
    model = build_detector(model_cfg['num_classes']).to(dev)
    try:
        state = torch.load(ckpt, map_location=dev, weights_only=True)['state_dict']  
    except TypeError:
        state = torch.load(ckpt, map_location=dev)['state_dict']
    model.load_state_dict(state)

    vlosses = []
    with torch.no_grad():
        model.train()
        for images, targets in tqdm(dl, desc="[det][eval] (loss)"):
            images = [im.to(dev) for im in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            vlosses.append(float(sum(loss_dict.values())))
    print(f"[det] val loss={float(np.mean(vlosses)) if vlosses else 0.0:.4f}")

    all_preds, all_gts = [], []
    with torch.no_grad():
        model.eval()
        for images, targets in tqdm(dl, desc=f"[det][eval] (mAP@0.5, thr={score_thr})"):
            images = [im.to(dev) for im in images]
            outputs = model(images)
            filtered = []
            for o in outputs:
                if len(o.get("scores", torch.empty(0))) == 0:
                    filtered.append({"boxes": np.empty((0,4), np.float32), "scores": np.empty((0,), np.float32)})
                    continue
                keep = (o["scores"] >= float(score_thr)).detach().cpu().numpy()
                boxes  = o["boxes"].detach().cpu().numpy()[keep]
                scores = o["scores"].detach().cpu().numpy()[keep]
                filtered.append({"boxes": boxes, "scores": scores})
            preds = filtered
            gts = [{"boxes": t.get("boxes", torch.empty(0,4)).detach().cpu().numpy()} for t in targets]
            all_preds.extend(preds); all_gts.extend(gts)
    mAP, P, R = evaluate_map_at_05(all_preds, all_gts)
    print(f"[det] mAP@0.5={mAP:.4f}  Precision={P:.4f}  Recall={R:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['seg','det'], required=True)
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--score_thr', type=float, default=0.05, help='Detection confidence threshold')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.task == 'seg':
        eval_seg(cfg, args.ckpt)
    else:
        eval_det(cfg, args.ckpt, score_thr=args.score_thr)

if __name__ == '__main__':
    main()


