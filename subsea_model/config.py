from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

from .model import UnderWaterModelConfig
from .utils import get_script_dir
from .optimisers import RMSPropOptimizerConfig, SDGOptimizerConfig

import os

CONFIG_FILE_PATH='subsea_model\\config_file.txt'

def load_config(file_name, config):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parameter = line.split(',')[0]
            value     = line.split(',')[1].strip()
            if parameter == 'learning_rate':
                config['lr'] = float(value)
            elif parameter == 'optimiser':
                if value == 'RMSProp':
                    config['optimiser'] = RMSPropOptimizerConfig(lr=config['lr'], eps=1e-15)
                elif value == 'SGD':
                    config['optimiser'] = SDGOptimizerConfig(lr=config['lr'], eps=1e-15)
                elif value == 'Adam':
                    config['optimiser'] = AdamOptimizerConfig(lr=config['lr'], eps=1e-15)
                elif value == 'RAdam':
                    config['optimiser'] = RAdamOptimizerConfig(lr=config['lr'], eps=1e-15)
            elif parameter == 'downscaling':
                if value == '1':
                    config['downscaling'] = None
                else:
                    config['downscaling'] = int(value)
            elif parameter == 'base_value':
                config['base_value'] = float(value)
            elif parameter in ['grad_scaled_loss', 'use_depth_renderer']:
                config[parameter] = value == 'True'
            else:
                config[parameter] = value
# Default configuration parameters
config = {
    'lr': 0.01,
    'optimiser': RAdamOptimizerConfig(lr=1e-2, eps=1e-15),
    'downscaling': None,
    'base_value': 0.0,
    'rgb_loss_type': 'Normal',
    'grad_scaled_loss': False,
    'use_depth_renderer': True
}

# Load configuration if config file exists
if os.path.exists(CONFIG_FILE_PATH):
    load_config(CONFIG_FILE_PATH, config)

# Base method configuration
SubseaModel = MethodSpecification(
    config=TrainerConfig(
        method_name="subsea-model",
        steps_per_eval_batch=500,
        steps_per_eval_all_images=10000,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=ParallelDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(downscale_factor=config['downscaling']),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
            ),
            model=UnderWaterModelConfig(
                eval_num_rays_per_chunk=1 << 15, 
                average_init_density=0.1,
                camera_optimizer=CameraOptimizerConfig(mode="SO3xR3"),
                depth_renderer_distribution_base=config['base_value'],
                rgb_loss_type=config['rgb_loss_type'],
                use_gradient_scaling=config['grad_scaled_loss'],
                use_depth_renderer=config['use_depth_renderer']
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": config['optimiser'],
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=50000, warmup_steps=1024),
            },
            "fields": {
                "optimizer": config['optimiser'],
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=50000, warmup_steps=1024),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRF for underwater scenes.",
)
