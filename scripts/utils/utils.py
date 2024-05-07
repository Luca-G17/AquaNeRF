import cv2
import numpy as np
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity
from torch import from_numpy, clamp

def get_images_in_dir(dir):
    image_exts = ['.png', '.jpg']
    files = os.listdir(dir)
    files = [f for f in files if any(f.endswith(ext) for ext in image_exts)]
    return files

def psnr(original, reconstruction):
    if (original.shape != reconstruction.shape):
        h, w, _ = reconstruction.shape
        original = cv2.resize(original, (w, h))
        print(f"PSNR: Input image shapes do not match, resizing to {(w, h)}")
    mse = np.mean((original.astype(np.float64) / 255 - reconstruction.astype(np.float64) / 255) ** 2)
    if (mse == 0):
        psnr = 100
    else:
        psnr = 10 * np.log10(1. / mse)
    return psnr

def ssim(original, reconstruction):
    score = structural_similarity(original, reconstruction, channel_axis=-1)
    return score

def lpips(original, reconstruction, model):
    t_original = clamp(from_numpy(np.transpose([original], (0, 3, 1, 2))) / 255.0, 0, 1)
    t_reconstruction = clamp(from_numpy(np.transpose([reconstruction], (0, 3, 1, 2))) / 255.0, 0, 1)
    return model(t_original, t_reconstruction)