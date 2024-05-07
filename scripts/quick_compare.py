import cv2
import argparse
from utils.utils import ssim, psnr

def quick_eval(image1_name, image2_name):
    image1 = cv2.imread(image1_name)
    image2 = cv2.imread(image2_name)
    s = ssim(image1, image2)
    p = psnr(image1, image2)
    print(f"SSIM: {s}")
    print(f"PSNR: {p}")

parser = argparse.ArgumentParser()
parser.add_argument('--path1')
parser.add_argument('--path2')
args = parser.parse_args()
if args.path1 != '' and args.path2 != '':
    quick_eval(args.path1, args.path2)