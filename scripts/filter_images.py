import os
import shutil

def filter(scaling=3, dir="./Video3/"):
    files = os.listdir(dir)
    files = [ file for file in files if '.png' in file]
    files = sorted(files, key=lambda f: int(f.split('.png')[0]))
    dest = dir + 'filtered/'
    if not os.path.exists(dest):
        os.mkdir(dest)

    f = 1
    for i in range(0, len(files), scaling):
        shutil.copy(dir + files[i], dest + f'{f:04d}.png')
        f += 1

filter()