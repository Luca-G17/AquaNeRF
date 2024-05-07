import os
import time
from psnr_evaluation import average_metrics

def write_param_to_config(parameter, value, config_file='subsea_model/config_file.txt'):
    with open(config_file, 'r') as f:
        lines = f.readlines()
        i = next((ind for ind, line in enumerate(lines) if line.split(',')[0] == parameter), None)
    if i != None:
        lines[i] = f'{parameter}, {value}\n'
        with open(config_file, 'w') as f:
            f.writelines(lines)

def experiment_runner(dataset_dir, exp_name, params=[], model='subsea-model', quit_on_completion=True, render_only=False, render_base=''):
    for (parameter, value) in params:
        write_param_to_config(parameter, value)

    if not render_only:
        print(f'[AUTO-EXPERIMENT-RUNNER] -- Running {exp_name}')
        os.system(f'ns-train {model} --vis=viewer+wandb --data {dataset_dir} --experiment_name={exp_name} --viewer.quit-on-train-completion={"True" if quit_on_completion else "False"}')
    
    print(f'[AUTO-EXPERIMENT-RUNNER] -- Rendering {exp_name}')
    os.system(f'python scripts/render.py --experiment_name={exp_name} --method=eval --base_render_dir={render_base}')

def learning_rate_experiment(dataset_dir, downscaling=1, run=True):
    lrs = ['0.01', '0.001', '0.1']
    names = []
    write_param_to_config('downscaling', downscaling)
    for lr in lrs:
        exp_name = f'subsea_{dataset_dir.split("/")[-1]}_lr_{int(float(lr) * 1000)}'
        names.append(exp_name)
        if run:
            experiment_runner(dataset_dir, exp_name, [['learning_rate', lr]])
    return names

def base_value_experiment(dataset_dir, downscaling=1, run=True):
    bases = ['0.0', '0.1', '0.2', '0.3']
    names = []
    write_param_to_config('downscaling', downscaling)
    for base in bases:
        exp_name = f'subsea_{dataset_dir.split("/")[-1]}_base_{int(float(base) * 1000)}'
        names.append(exp_name)
        if run:
            experiment_runner(dataset_dir, exp_name, [['base_value', base]])
    return names

def dataset_reduction_experiment(dataset_dir, downscaling=1, run=True):
    proportion = ['30', '50', '70', '90']
    names = []
    write_param_to_config('downscaling', downscaling)
    for prop in proportion:
        exp_name = f'subsea_{dataset_dir.split("/")[-1]}_prop_{prop}'
        dataset_dir_ext = f'{dataset_dir}\\filtered_psnr_{prop}'
        names.append(exp_name)
        if run:
            experiment_runner(dataset_dir_ext, exp_name)

    return names

def optimiser_experiment(dataset_dir, run=True):
    optimisers = ['SGD', 'RMSProp', 'Adam', 'RAdam']
    names = []
    for optimiser in optimisers:
        exp_name = f'subsea_{dataset_dir.split("/")[-1]}_{optimiser}'
        names.append(exp_name)
        if run:
            experiment_runner(dataset_dir, exp_name, [['optimiser', optimiser]])
    return names

def eval_performance(exp_name):
    gt_rgb = f'renders/{exp_name}/eval/val/gt-rgb'
    rgb    = f'renders/{exp_name}/eval/val/rgb'
    return average_metrics(gt_rgb, rgb, True)

def collate_eval_data(experiment_names):
    print(f'|-------------------------------|-------|------|-------|')
    print(f'| {"Experiment Name": <30}| PSNR  | SSIM | LPIPS |')
    print(f'|-------------------------------|-------|------|-------|')
    for name in experiment_names:
        data = eval_performance(name)
        print(f'| {name: <30}| {data["PSNR"]:.2f} | {data["SSIM"]:.2f} | {data["LPIPS"]:.2f}  |')
    print(f'|-------------------------------|-------|------|-------|')


def reset_config_to_defaults():
    defaults = {
        'downscaling': 1,
        'learning_rate': 0.01,
        'optimiser': 'RAdam',
        'base_value': 0.1,
        'rgb_loss_type': 'Normal',
        'grad_scaled_loss': 'False',
        'use_depth_renderer': 'True'
    }
    for param, value in defaults.items():
        write_param_to_config(param, value)

class Experiment:
    def __init__(self, exp_name, dataset_dir, model='subsea-model', params=[], base_render_dir=''):
        self.exp_name = exp_name
        self.dataset_dir = dataset_dir
        self.params = params
        self.model = model
        self.base_render_dir = base_render_dir

def process_experiment_queue(exps, quit_on_completion=True, render_only=False):
    results_dir = 'experiment_results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    now = int(time.time())
    exp_result_dir = f'{results_dir}\\{now}'
    os.mkdir(exp_result_dir)

    results_file = f'{exp_result_dir}\\result.out'
    with open(results_file, 'w') as f:
        f.write(f'AUTO-EXPERIMENT RUNNER RESULTS:\n')
        f.write(f'\n')
        f.write(f'|-----------------------------------------|-------|------|-------|\n')
        f.write(f'| {"Experiment Name": <40}| PSNR  | SSIM | LPIPS |\n')
        f.write(f'|-----------------------------------------|-------|------|-------|\n')

    for experiment in exps:
        reset_config_to_defaults()
        experiment_runner(
                        dataset_dir=experiment.dataset_dir,
                        exp_name=experiment.exp_name,
                        params=experiment.params,
                        model=experiment.model,
                        quit_on_completion=quit_on_completion,
                        render_only=render_only,
                        render_base=experiment.base_render_dir)
        data = eval_performance(experiment.exp_name)
        with open(results_file, 'a') as f:
            f.write(f'| {experiment.exp_name: <40}| {data["PSNR"]:.2f} | {data["SSIM"]:.2f} | {data["LPIPS"]:.2f}  |\n')

    with open(results_file, 'a') as f:
        f.write(f'|-----------------------------------------|-------|------|-------|\n')
                
    return [exp.exp_name for exp in exps]

# names = optimiser_experiment('Datasets/original_colmap', False)
# names = dataset_reduction_experiment('Datasets/video2_colmap', downscaling=8, run=False)
# names = learning_rate_experiment('Datasets/video2_colmap', run=False)
# names = base_value_experiment('Datasets/video3_colmap')

reset_config_to_defaults()
exps = [
    Experiment(exp_name='nerfacto_vid3', dataset_dir='Datasets/video3_colmap', model='nerfacto'),
    Experiment(exp_name='vid1_render_robust', dataset_dir='Datasets/original_colmap', params=[['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='vid2_render_robust', dataset_dir='Datasets/video2_colmap', params=[['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='nerfacto_shipwreck', dataset_dir='Datasets/shipwreck_colmap/filtered_7', model='nerfacto'),
    Experiment(exp_name='shipwreck_render_robust', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='vid1_render_robust_grad', dataset_dir='Datasets/original_colmap', params=[['rgb_loss_type', 'Robust'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='vid2_render_robust_grad', dataset_dir='Datasets/video2_colmap', params=[['rgb_loss_type', 'Robust'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='vid3_render_robust_grad', dataset_dir='Datasets/video3_colmap', params=[['rgb_loss_type', 'Robust'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='shipwreck_render_robust_grad', dataset_dir='Datasets/shipwreck_colmap', params=[['rgb_loss_type', 'Robust'], ['grad_scaled_loss', 'True']])
]


# TODO: Before you run these experiments add the 'use_depth_renderer' flag to the params file and add a corresponding handler to the model config
exps_2 = [
    Experiment(exp_name='vid1_robust', dataset_dir='Datasets/original_colmap', params=[['use_depth_renderer', 'False'], ['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='vid2_robust', dataset_dir='Datasets/video2_colmap', params=[['use_depth_renderer', 'False'], ['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='vid3_robust', dataset_dir='Datasets/video3_colmap', params=[['use_depth_renderer', 'False'], ['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='shipwreck_robust', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['use_depth_renderer', 'False'], ['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='vid1_grad', dataset_dir='Datasets/original_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='vid2_grad', dataset_dir='Datasets/video2_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]), 
    Experiment(exp_name='vid3_grad', dataset_dir='Datasets/video3_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='shipwreck_grad', dataset_dir='Datasets/shipwreck_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]),
]

exps_3 = [
    Experiment(exp_name='vid1_grad', dataset_dir='Datasets/original_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='vid2_grad', dataset_dir='Datasets/video2_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]), 
    Experiment(exp_name='vid3_grad', dataset_dir='Datasets/video3_colmap', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='shipwreck_grad', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['use_depth_renderer', 'False'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='shipwreck_robust', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['use_depth_renderer', 'False'], ['rgb_loss_type', 'Robust']]),
    Experiment(exp_name='nerfacto_shipwreck', dataset_dir='Datasets/shipwreck_colmap/filtered_7', model='nerfacto'),
    Experiment(exp_name='shipwreck_render_robust', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['rgb_loss_type', 'Robust']]),
]

exps_4 = [
   #  Experiment(exp_name='shipwreck_render_grad', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['grad_scaled_loss', 'True']]),
   # Experiment(exp_name='shipwreck_render_robust_grad', dataset_dir='Datasets/shipwreck_colmap/filtered_7', params=[['rgb_loss_type', 'Robust'], ['grad_scaled_loss', 'True']]),
    Experiment(exp_name='shipwreck_nerfacto', dataset_dir='Datasets/shipwreck_colmap/filtered_7', model='nerfacto'),
    Experiment(exp_name='vid1_seathru', dataset_dir='Datasets/original_colmap', model='seathru-nerf'),
    Experiment(exp_name='vid2_seathru', dataset_dir='Datasets/video2_colmap', model='seathru-nerf'),
    Experiment(exp_name='vid3_seathru', dataset_dir='Datasets/video3_colmap', model='seathru-nerf'),
]

exps_5 = [
    Experiment(exp_name='dynamic_object_3', dataset_dir='Datasets/dynamic_object_3_colmap'),
    Experiment(exp_name='dynamic_object_3_robust', dataset_dir='Datasets/dynamic_object_3_colmap', params=[['rgb_loss_type', 'Robust']])
]

names = process_experiment_queue(exps_5, quit_on_completion=True)
collate_eval_data(names)

