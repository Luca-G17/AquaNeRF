import os
import argparse
import shutil

RENDERS = './renders'

def read_ext(frame_dir, frame_no):
    frames = os.listdir(frame_dir)
    for frame in frames:
        if f'{frame_no:05}' in frame:
            return frame.split('.')[1]

def extract_images(dataset, frame_no, ext='jpg', base_dir=''):
    dataset_synonyms = [dataset]
    if dataset == 'vid1':
        dataset_synonyms.append('original')
    elif dataset == 'vid2':
        dataset_synonyms.append('video2')

    experiments = [dir for dir in os.listdir(f"RENDERS/{base_dir}") if any([synonym in dir for synonym in dataset_synonyms])]
    experiments.append(experiments[0])
    isolated_dir = f'{RENDERS}/{base_dir}isolated_frames/{dataset}/{frame_no:05}'
    if not os.path.exists(isolated_dir):
        os.makedirs(isolated_dir)
    
    for i, exp in enumerate(experiments):
        gt = ''
        if i == len(experiments) - 1:
            gt = 'gt-'
        
        frame_dir = f'{RENDERS}/{base_dir}{exp}/eval/val/{gt}rgb'
        ext = read_ext(frame_dir, frame_no)
        frame = f'{frame_dir}/frame_{frame_no:05}.{ext}'
        if os.path.exists(frame):
            if i == len(experiments) - 1:
                exp = 'gt-rgb'
            shutil.copy(frame, f'{isolated_dir}/{exp}.{ext}')
            print(f'Copied From: {exp}')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--frame_no', type=int)
parser.add_argument('--ext', type=str)
args = parser.parse_args()
if args.ext == '':
    args.ext = '.jpg'

extract_images(args.dataset, args.frame_no, args.ext, base_dir='png_versions/')