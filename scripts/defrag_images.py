import os

def shift_images(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    index = 1
    for filename in files:
        name, extension = os.path.splitext(filename)
        frame_number = int(name.split('_')[1])
        if (frame_number != index):
            shifted = f'frame_{index:05d}{extension}'
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, shifted))
        index += 1

folder_path = 'train'
shift_images(folder_path)