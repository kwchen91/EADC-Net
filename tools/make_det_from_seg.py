from pathlib import Path
import json, re
import numpy as np
from PIL import Image

ROOT = Path(".")
IMG_DIR = ROOT / "toy_seg" / "images"
MSK_DIR = ROOT / "toy_seg" / "masks"
DET_IMG = ROOT / "toy_det" / "images"
DET_LBL = ROOT / "toy_det" / "labels"
DET_IMG.mkdir(parents=True, exist_ok=True)
DET_LBL.mkdir(parents=True, exist_ok=True)

def bin_mask(m: Image.Image):
    arr = np.array(m.convert("L"))
    return (arr > 127).astype(np.uint8)

def cc_boxes(bin_arr: np.ndarray):
    H, W = bin_arr.shape
    vis = np.zeros_like(bin_arr, bool)
    out=[]; nbrs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for y in range(H):
        for x in range(W):
            if bin_arr[y,x]==0 or vis[y,x]: continue
            st=[(y,x)]; vis[y,x]=True; ys=[]; xs=[]
            while st:
                cy,cx=st.pop(); ys.append(cy); xs.append(cx)
                for dy,dx in nbrs:
                    ny,nx=cy+dy, cx+dx
                    if 0<=ny<H and 0<=nx<W and not vis[ny,nx] and bin_arr[ny,nx]==1:
                        vis[ny,nx]=True; st.append((ny,nx))
            ymin,ymax=min(ys),max(ys); xmin,xmax=min(xs),max(xs)
            pad=max(2,int(0.02*max(H,W)))
            xmin=max(0,xmin-pad); ymin=max(0,ymin-pad)
            xmax=min(W-1,xmax+pad); ymax=min(H-1,ymax+pad)
            out.append((xmin,ymin,xmax,ymax))
    return out

def main():
    rows=[]
    for ip in sorted(IMG_DIR.glob("*.png")):
        mp = MSK_DIR / ip.name
        if not mp.exists(): continue
        im = Image.open(ip).convert("L")
        im.save(DET_IMG / ip.name)
        arr = bin_mask(Image.open(mp))
        H,W = arr.shape
        boxes = cc_boxes(arr)
        ytxt = DET_LBL / f"{ip.stem}.txt"
        lines=[]
        for x1,y1,x2,y2 in boxes:
            xc=((x1+x2)/2)/W; yc=((y1+y2)/2)/H; bw=(x2-x1)/W; bh=(y2-y1)/H
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        ytxt.write_text("\n".join(lines), encoding="utf-8")
    (ROOT/"toy_det"/"classes.txt").write_text("vertebra\n", encoding="utf-8")
    print("done.")
if __name__ == "__main__":
    main()
