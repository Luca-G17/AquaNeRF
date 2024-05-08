import cv2
import numpy as np
import os
from math import log10, sqrt
from skimage.metrics import structural_similarity
from torch import from_numpy, clamp
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import re

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
    score = structural_similarity(original, reconstruction, channel_axis=-1, multichannel=True)
    return score

def lpips(original, reconstruction, model):
    t_original = clamp(from_numpy(np.transpose([original], (0, 3, 1, 2))) / 255.0, 0, 1)
    t_reconstruction = clamp(from_numpy(np.transpose([reconstruction], (0, 3, 1, 2))) / 255.0, 0, 1)
    return model(t_original, t_reconstruction)


def average_metrics(original_dir, reconstruction_dir, suppress_output=False):
    recons = get_images_in_dir(reconstruction_dir)
    originals = get_images_in_dir(original_dir)

    image_dict = {}
    for image in originals:
        image_no = re.findall(r"[0-9]+", image)[0]
        image_dict[image_no] = [image]
    for image in recons:
        image_no = re.findall(r"[0-9]+", image)[0]
        image_dict[image_no].append(image)
    
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    lpips_model = LearnedPerceptualImagePatchSimilarity(normalize=True)
    for i, image_pair in enumerate(image_dict.values()):
        original = cv2.imread(f'{original_dir}/{image_pair[0]}')
        reconstruction = cv2.imread(f'{reconstruction_dir}/{image_pair[1]}')
        total_psnr += psnr(original, reconstruction)
        total_ssim += ssim(original, reconstruction)
        total_lpips += lpips(original, reconstruction, lpips_model)
        if not suppress_output:
            print(f"Computing metrics (PSNR+SSIM+LPIPS) for images: {i + 1:03}/{len(image_dict.values())}\r", end="")
    
    average_psnr = total_psnr / float(len(image_dict.values()))
    average_ssim = total_ssim / float(len(image_dict.values()))
    average_lpips = total_lpips / float(len(image_dict.values()))
    if not suppress_output:
        print()
        print(f"Average PSNR: {average_psnr:.2f}dB")
        print(f"Average SSIM: {average_ssim:.2f}")
        print(f"Average LPIPS: {average_lpips:.2f}")
    return { 'PSNR': average_psnr, 'SSIM': average_ssim, 'LPIPS': average_lpips }