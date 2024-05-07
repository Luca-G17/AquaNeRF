import cv2
import os
import argparse
import re
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils.utils import psnr, ssim, lpips, get_images_in_dir

RENDERS='renders/'

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

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', default='')
parser.add_argument('--recon_path', default='')
args = parser.parse_args()
if args.gt_path != '' and args.recon_path != '':
    average_metrics(args.gt_path, args.recon_path)
    
