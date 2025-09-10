import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def load_history(path: Path):
    xs, ys = [], []
    if not path.exists():
        print(f"[warn] no history file: {path}")
        return xs, ys
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            xs.append(rec.get('epoch', 0))
            # seg: val_dice; det: val_loss
            val = rec.get('val_dice', None)
            if val is None:
                val = rec.get('val_loss', None)
            ys.append(val)
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['seg','det'], required=True)
    ap.add_argument('--runs', type=str, default='runs')
    args = ap.parse_args()

    runs = Path(args.runs) / args.task
    hist = runs / 'history.jsonl'
    xs, ys = load_history(hist)
    if not xs:
        print("[warn] empty history; did you add logging in train.py?")
        return

    plt.figure()
    if args.task == 'seg':
        plt.plot(xs, ys, marker='o'); plt.ylabel('Val Dice'); plt.xlabel('Epoch'); plt.title('Segmentation Val Dice')
    else:
        plt.plot(xs, ys, marker='o'); plt.ylabel('Val Loss'); plt.xlabel('Epoch'); plt.title('Detection Val Loss')
    out = runs / 'curve.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"[ok] saved {out}")

if __name__ == '__main__':
    main()
