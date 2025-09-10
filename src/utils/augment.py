import random
import torchvision.transforms.functional as TF

def aug_seg(img, mask):
    if random.random() < 0.5:
        img = TF.hflip(img); mask = TF.hflip(mask)
    if random.random() < 0.5:
        img = TF.vflip(img); mask = TF.vflip(mask)
    return img, mask

def aug_det(img):
    if random.random() < 0.5:
        img = TF.hflip(img)
    return img