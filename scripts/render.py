from os import listdir, makedirs, system
from os.path import isfile, join
from datetime import datetime
import argparse
import numpy as np

OUTPUTS_DIR = "./outputs/"
RENDERS_DIR = "./renders/"

def render(experiment_name="subsea_far_plane", method="interpolate", base_render_dir=''):
    dir = OUTPUTS_DIR + experiment_name
    model_name = listdir(dir)[0]

    # outputs/[EXPERIMENT_NAME]/[MODEL]/[date]_[time]/
    exps = [ join(dir + "/" + model_name, f) for f in listdir(dir + "/" + model_name) ]
    if len(exps) == 0:
        print(f"Experiment {experiment_name} not found")
        return
    
    dates = [datetime.strptime(d.strip(dir + "/" + model_name).strip("\\"), "%Y-%m-%d_%H%M%S") for d in exps]
    exp = exps[np.argmax(dates)].replace("\\", "/")

    x = 1
    experiment_name_cpy = experiment_name
    while experiment_name_cpy in listdir(RENDERS_DIR) and method == "interpolate":
        experiment_name_cpy = experiment_name + f'({x})'
        x += 1
    
    render_dir = RENDERS_DIR + base_render_dir + experiment_name_cpy
    if method == "eval":
        method = "dataset --split val"
        render_dir += "/eval"
    elif method == "interpolate":
        method += " --output-format images"

    makedirs(render_dir)
    system(f"ns-render {method} --output-path {render_dir} --load-config {exp + '/config.yml'} --image-format=png")  
    if method != "dataset --split val":
        system(f"ffmpeg -framerate 24 -i {render_dir}/%05d.jpg -vf scale='950:550' {render_dir}/output.mp4")
        print(f"Finished rendering video: {render_dir}/output.mp4")

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--method", type=str)
parser.add_argument("--base_render_dir", type=str)
args = parser.parse_args()
if args.method == None:
    args.method = "interpolate"
if args.base_render_dir == None:
    args.base_render_dir = ""
render(args.experiment_name, args.method, args.base_render_dir)