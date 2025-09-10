import streamlit as st
import torch, numpy as np
from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
import torchvision.transforms.functional as TF
from src.models.segmentation import build_unet
from src.models.detection import build_detector

st.set_page_config(page_title="EADC-Net Demo", layout="centered")
st.title("Lumbar X-ray Segmentation & Detection (Code-only Demo)")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tab1, tab2 = st.tabs(["Segmentation", "Detection"])

with tab1:
    ckpt = st.text_input("Segmentation checkpoint (.pt)", "runs/seg/best.pt")
    up = st.file_uploader("Upload an image (png/jpg)", type=["png","jpg","jpeg"], key="seg")
    if st.button("Run segmentation") and up:
        img = Image.open(up).convert("L").resize((512,512))
        x = TF.to_tensor(img).unsqueeze(0).to(device)
        m = build_unet(1,1).to(device).eval()
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state['state_dict'])
        with torch.no_grad():
            prob = torch.sigmoid(m(x))[0,0].cpu().numpy()
        mask = (prob>0.5).astype(np.uint8)*255
        over = ImageOps.colorize(Image.fromarray(mask), black=(0,0,0), white=(0,255,0)).convert("RGBA")
        out = Image.blend(img.convert("RGBA"), over, 0.35).convert("RGB")
        st.image(out, use_column_width=True)

with tab2:
    ckpt = st.text_input("Detection checkpoint (.pt)", "runs/det/best.pt")
    thr = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.01)
    up = st.file_uploader("Upload an image (png/jpg)", type=["png","jpg","jpeg"], key="det")
    if st.button("Run detection") and up:
        img = Image.open(up).convert("L").resize((640,640))
        x = TF.to_tensor(img).repeat(3,1,1).unsqueeze(0).to(device)
        m = build_detector(2).to(device).eval()
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state['state_dict'])
        with torch.no_grad():
            out = m(x)[0]
        keep = out['scores'].cpu().numpy() >= thr
        boxes = out['boxes'].cpu().numpy()[keep]
        rgb = img.convert("RGB")
        dr = ImageDraw.Draw(rgb)
        for x1,y1,x2,y2 in boxes:
            dr.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
        st.image(rgb, use_column_width=True)
