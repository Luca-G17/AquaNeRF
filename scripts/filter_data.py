import os
import cv2 
import argparse
import json
import shutil
from utils.utils import psnr

RENDERS='renders/'

def copy_images(source_dir, destination, frame_names, total_copied):
    for im in frame_names:
        total_copied += 1
        shutil.copyfile(f'{source_dir}/{im}', f'{destination}/{im}')
        print(f'Copying images: {total_copied + 1}/{4 * int(len(frame_names))}\r', end="")

    return total_copied

def estimate_psnr_proportions(dataset_dir):
    with open(f'{dataset_dir}/transforms.json', 'r') as t_file:
        transforms = json.load(t_file)
        frame_count = len(transforms['frames'])
        frames = transforms['frames']
        frames = sorted(frames, key=lambda f : int(f['file_path'].replace('images/frame_', '').replace('.jpg', '').replace('.png', '')))

    estimator_split = 0.05
    average_neighbor_psnr = 0
    count = 0
    for i in range(0, frame_count - 1, int(1.0 / estimator_split)):
        image1 = cv2.imread(f'{dataset_dir}/{frames[i]["file_path"]}')
        image2 = cv2.imread(f'{dataset_dir}/{frames[i + 1]["file_path"]}')
        average_neighbor_psnr += psnr(image1, image2)
        count += 1
        
    average_neighbor_psnr = average_neighbor_psnr / count
    upper = 100 * filter_images(dataset_dir, filter_regime='psnr', psnr_threshold=average_neighbor_psnr, get_props=True)
    lower = 100 * filter_images(dataset_dir, filter_regime='psnr', psnr_threshold=average_neighbor_psnr - 4, get_props=True)
    gradient = 4.0 / (upper - lower)
    c        = average_neighbor_psnr - (gradient * upper)
    print(f'PSNR Threshold = {average_neighbor_psnr} ---> Kept {upper:.2f}%')
    print(f'PSNR Threshold = {average_neighbor_psnr - 4} ---> Kept {lower:.2f}%')
    print(f'Extrapolating...')
    print()
    print(f'| % of dataset retained | PSNR Threshold ')
    print(f'|-----------------------|----------------')
    print(f'| 30%                   | {gradient * 30 + c:.2f}')
    print(f'| 50%                   | {gradient * 50 + c:.2f}')
    print(f'| 70%                   | {gradient * 70 + c:.2f}')
    print(f'| 90%                   | {gradient * 90 + c:.2f}')
    

def filter_images(dataset_dir, filter_regime='proportion', keep_proportion=0.7, psnr_threshold=28.0, get_props=False):
    if filter_regime == 'proportion':
        filtered_dir = f'{dataset_dir}/filtered_prop_{int(keep_proportion * 10)}'
    elif filter_regime == 'psnr':
        filtered_dir = f'{dataset_dir}/filtered_psnr_{int(psnr_threshold * 10)}'

    if not os.path.exists(filtered_dir):
        os.mkdir(filtered_dir)
        os.mkdir(f'{filtered_dir}/images')
        os.mkdir(f'{filtered_dir}/images_2')
        os.mkdir(f'{filtered_dir}/images_4')
        os.mkdir(f'{filtered_dir}/images_8')
    
    with open(f'{dataset_dir}/transforms.json', 'r') as t_file:
        transforms = json.load(t_file)
        remove_every = int(1 / (1 - keep_proportion))
        kept_frames = []
        frame_count = len(transforms['frames'])
        frames = transforms['frames']
        frames = sorted(frames, key=lambda f : int(f['file_path'].replace('images/frame_', '').replace('.jpg', '').replace('.png', '')))

        if filter_regime == 'proportion':
            for i in range(frame_count):
                if (i + 1) % remove_every != 0:
                    kept_frames.append(frames[i])
        elif filter_regime == 'psnr':
            i = 0
            while i < frame_count:
                kept_frames.append(frames[i])
                offset = 0
                f = i
                psnr_value = 100
                frame1 = cv2.imread(f'{dataset_dir}/{frames[i]["file_path"]}')
                while psnr_value > psnr_threshold and f < frame_count - 1:
                    offset += 1
                    f = i + offset
                    frame2 = cv2.imread(f'{dataset_dir}/{frames[f]["file_path"]}')
                    psnr_value = psnr(frame1, frame2)
                    print(f'Processing PSNR comparison: {f}/{frame_count}\r', end='')
                if offset == 0: i += 1 
                else: i += offset             

        transforms['frames'] = kept_frames

    if (get_props):
        return len(kept_frames) / frame_count
 
    with open(f'{filtered_dir}/transforms.json', 'w', encoding='utf-8') as f:
        json.dump(transforms, f, ensure_ascii=False, indent=4)

    print(f'Finished writing {filtered_dir}/transforms.json')
    kept_frame_names = [frame['file_path'].replace('images/', '') for frame in kept_frames]
    total_copied = 0
    total_copied = copy_images(f'{dataset_dir}/images', f'{filtered_dir}/images', kept_frame_names, total_copied)
    total_copied = copy_images(f'{dataset_dir}/images_2', f'{filtered_dir}/images_2', kept_frame_names, total_copied)
    total_copied = copy_images(f'{dataset_dir}/images_4', f'{filtered_dir}/images_4', kept_frame_names, total_copied)
    total_copied = copy_images(f'{dataset_dir}/images_8', f'{filtered_dir}/images_8', kept_frame_names, total_copied)
    print(f'Kept {100 * len(kept_frame_names) / frame_count:.2f}% of the original dataset -- ({len(kept_frame_names)}/{frame_count})')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--path")
    parser.add_argument("--filter_regime", default='proportion')
    parser.add_argument("--psnr_threshold", default=280, type=int)
    args = parser.parse_args()
    if args.path != '':
        if args.estimate:
            estimate_psnr_proportions(args.path)
        else:
            filter_images(args.path, filter_regime=args.filter_regime, psnr_threshold=args.psnr_threshold / 10.0)

if __name__ == '__main__':
    main()