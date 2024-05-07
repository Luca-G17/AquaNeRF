import cv2
import numpy as np
import os
import argparse

RENDERS='renders/'

def unmask_sharpen(image, scalar=1.5, threshold=0):
    blurred = cv2.GaussianBlur(image, (3, 3), 1)
    sharpend = float(scalar + 1) * image - float(scalar) * blurred
    sharpend = np.maximum(sharpend, np.zeros(sharpend.shape))
    sharpend = np.minimum(sharpend, np.ones(sharpend.shape) * 255)
    sharpend = sharpend.round().astype(np.uint8)
    if threshold > 0:
        low_constrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpend, image, where=low_constrast_mask)
    return sharpend

def sharpen_folder(folder, scalar=1.5):
    img_ext = [".png", ".jpg"]
    files = os.listdir(folder)
    files = [f for f in files if any(f.endswith(ext) for ext in img_ext)]
    sharpend_folder = f'{folder}/sharpend_{str(scalar).replace(".", "")}/'
    if not os.path.exists(sharpend_folder):
        os.mkdir(sharpend_folder)

    for i, image_name in enumerate(files):
        image = cv2.imread(f'{folder}/{image_name}')
        sharpend = unmask_sharpen(image, scalar)
        cv2.imwrite(f'{sharpend_folder}/{image_name}', sharpend)
        print(f'sharpend images: {i + 1:03}/{len(files)}\r', end="")

parser = argparse.ArgumentParser()
parser.add_argument('--frame_dir', type=str, default='')
parser.add_argument('--scalar', type=int, default=15)
args = parser.parse_args()
if args.frame_dir == '':
    print("Command line argument 'frame_dir' missing")
else:
    sharpen_folder(RENDERS + args.frame_dir, args.scalar / float(10))
