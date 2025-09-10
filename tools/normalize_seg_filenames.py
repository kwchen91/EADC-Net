from pathlib import Path
import re
from PIL import Image

ROOT = Path(".")

def collapse_double_ext(p: Path) -> Path:
    name = re.sub(r"\.(png|jpg|jpeg|tif|tiff|bmp)\.(png|jpg|jpeg|tif|tiff|bmp)$", r".\1", p.name, flags=re.I)
    return p.with_name(name)

def fix_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for p in list(folder.glob("*")):
        if not p.is_file(): continue
        q = collapse_double_ext(p)
        if q.name != p.name:
            if q.exists(): p.unlink()
            else: p.rename(q)
    for p in list(folder.glob("*")):
        if p.suffix.lower() != ".png" and p.is_file():
            im = Image.open(p).convert("L")
            im.save(p.with_suffix(".png")); p.unlink()

if __name__ == "__main__":
    for sub in ["toy_seg/images", "toy_seg/masks", "toy_det/images"]:
        fix_folder(ROOT / sub)
    print("normalize done.")
