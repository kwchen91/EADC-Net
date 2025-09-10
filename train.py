import json
import argparse, yaml, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from src.data.datasets import SegmentationDataset, DetectionDataset
from src.models.segmentation import build_unet
from src.models.detection import build_detector
from src.losses import bce_dice_loss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_device(cfg_device):
    if cfg_device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg_device

def det_collate(batch):
    return tuple(zip(*batch))

def _filter_nonempty_targets(images, targets):
    keep_imgs, keep_tgts = [], []
    for img, tgt in zip(images, targets):
        boxes = tgt.get("boxes", None)
        if boxes is not None and len(boxes) > 0:
            keep_imgs.append(img); keep_tgts.append(tgt)
    return keep_imgs, keep_tgts


# Seg 
def train_seg(cfg):
    device = get_device(cfg.get('device', 'auto'))
    data = cfg['data']; training = cfg['training']; model_cfg = cfg['model']

    root = data['root']
    files = sorted([p.name for p in (Path(root) / data['img_dir']).iterdir() if p.is_file()])
    kf = KFold(n_splits=int(data.get('folds', 5)), shuffle=True, random_state=cfg.get('seed', 42))
    save_dir = Path(training.get('save_dir', 'runs/seg')); save_dir.mkdir(parents=True, exist_ok=True)

    hist_path = save_dir / 'history.jsonl'
    if hist_path.exists(): hist_path.unlink()

    lr = float(training.get('lr', 5e-4))
    wd = float(training.get('weight_decay', 1e-4))
    bs = int(training.get('batch_size', 2))
    nw = int(training.get('num_workers', 0))
    epochs = int(training.get('epochs', 5))

    best_dice = -1.0; best_path = save_dir / 'best.pt'

    for fold, (tr, va) in enumerate(kf.split(files), 1):
        tr_files = [files[i] for i in tr]; va_files = [files[i] for i in va]
        ds_tr = SegmentationDataset(root, data['img_dir'], data['mask_dir'], data['mask_map'],
                                    data['img_size'], tr_files, augment=True)
        ds_va = SegmentationDataset(root, data['img_dir'], data['mask_dir'], data['mask_map'],
                                    data['img_size'], va_files, augment=False)
        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw)
        dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=nw)

        model = build_unet(model_cfg['in_channels'], model_cfg['out_channels']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(1, epochs + 1):
            model.train()
            running = []
            pbar = tqdm(dl_tr, desc=f"[seg][fold {fold}] epoch {epoch}")
            for img, mask in pbar:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                loss = bce_dice_loss(logits, mask, cfg['loss']['bce_weight'], cfg['loss']['dice_weight'])
                opt.zero_grad(); loss.backward(); opt.step()
                running.append(float(loss.detach().cpu()))
                pbar.set_postfix(loss=running[-1] if running else 0.0)

            train_loss = float(np.mean(running)) if running else 0.0

            # validation (Dice)
            model.eval()
            dices = []
            with torch.no_grad():
                for img, mask in dl_va:
                    img, mask = img.to(device), mask.to(device)
                    prob = torch.sigmoid(model(img))
                    pred = (prob > 0.5).float()
                    inter = (pred * mask).sum().item()
                    den = pred.sum().item() + mask.sum().item() + 1e-6
                    dices.append(2 * inter / den)
            md = float(np.mean(dices)) if dices else 0.0

            if md > best_dice:
                best_dice = md
                torch.save({'state_dict': model.state_dict(), 'fold': fold, 'dice': best_dice}, best_path)

            with hist_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'fold': int(fold), 'epoch': int(epoch),
                                    'train_loss': train_loss, 'val_dice': md}) + '\n')

    print(f"[seg] best dice={best_dice:.4f}, saved {best_path}")


# Det 
def train_det(cfg):
    device = get_device(cfg.get('device', 'auto'))
    data = cfg['data']; training = cfg['training']; model_cfg = cfg['model']

    root = data['root']
    files = sorted([p.name for p in (Path(root) / data['img_dir']).iterdir() if p.is_file()])
    kf = KFold(n_splits=int(data.get('folds', 5)), shuffle=True, random_state=cfg.get('seed', 42))
    save_dir = Path(training.get('save_dir', 'runs/det')); save_dir.mkdir(parents=True, exist_ok=True)

    hist_path = save_dir / 'history.jsonl'
    if hist_path.exists(): hist_path.unlink()

    lr = float(training.get('lr', 5e-4))
    wd = float(training.get('weight_decay', 1e-4))
    bs = int(training.get('batch_size', 2))
    nw = int(training.get('num_workers', 0))
    epochs = int(training.get('epochs', 5))

    best_loss = 1e9; best_path = save_dir / 'best.pt'

    for fold, (tr, va) in enumerate(kf.split(files), 1):
        tr_files = [files[i] for i in tr]; va_files = [files[i] for i in va]
        ds_tr = DetectionDataset(root, data['img_dir'], data.get('labels_csv', 'labels.csv'),
                                 data.get('yolo_dir', 'labels'), data['img_size'], tr_files, augment=True)
        ds_va = DetectionDataset(root, data['img_dir'], data.get('labels_csv', 'labels.csv'),
                                 data.get('yolo_dir', 'labels'), data['img_size'], va_files, augment=False)

        dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, collate_fn=det_collate, num_workers=nw)
        dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, collate_fn=det_collate, num_workers=nw)

        model = build_detector(model_cfg['num_classes']).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

        for epoch in range(1, epochs + 1):
            model.train()
            running = []
            pbar = tqdm(dl_tr, desc=f"[det][fold {fold}] epoch {epoch}")
            for images, targets in pbar:
                images = [im.to(device) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                images, targets = _filter_nonempty_targets(images, targets)
                if not images: 
                    continue
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                opt.zero_grad(); loss.backward(); opt.step()
                running.append(float(loss.detach().cpu()))
                pbar.set_postfix(loss=running[-1] if running else 0.0)

            train_loss = float(np.mean(running)) if running else 0.0

            # validation 
            vlosses = []
            with torch.no_grad():
                model.train()  
                for images, targets in dl_va:
                    images = [im.to(device) for im in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    images, targets = _filter_nonempty_targets(images, targets)
                    if not images:
                        continue
                    ldict = model(images, targets)
                    vlosses.append(float(sum(ldict.values())))
            mv = float(np.mean(vlosses)) if vlosses else 0.0

            if mv < best_loss:
                best_loss = mv
                torch.save({'state_dict': model.state_dict(), 'fold': fold, 'loss': best_loss}, best_path)

            with hist_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps({'fold': int(fold), 'epoch': int(epoch),
                                    'train_loss': train_loss, 'val_loss': mv}) + '\n')

    print(f"[det] best val loss={best_loss:.4f}, saved {best_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['seg', 'det'], required=True)
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get('seed', 42))

    if args.task == 'seg':
        train_seg(cfg)
    else:
        train_det(cfg)

if __name__ == '__main__':
    main()
