from pathlib import Path
imgs = {p.stem for p in Path("toy_seg/images").glob("*.png")}
msks = {p.stem for p in Path("toy_seg/masks").glob("*.png")}
print("Total images:", len(imgs), "masks:", len(msks))
print("Missing masks for:", sorted(imgs - msks))
print("Missing images for:", sorted(msks - imgs))