from dataclasses import dataclass, field
from typing import Dict, List, Type, Literal, Tuple

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.renderers import AccumulationRenderer, RGBRenderer, DepthRenderer
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import MSELoss, interlevel_loss, scale_gradients_by_distance_squared, distortion_loss
from nerfstudio.utils import colormaps
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)


from .field import UnderWaterField
from .renderers import UnderWaterRGBRenderer
from .losses import recon_loss, robust_loss, reweighting_rgb_loss_rectifier
from .renderers import UnderWaterDepthRenderer


@dataclass
class UnderWaterModelConfig(ModelConfig):
    """Subsea NeRF Config."""

    _target: Type = field(default_factory=lambda: UnderWaterModel)
    near_plane: float = 0.05
    """Near plane of rays."""
    far_plane: float = 3000
    """Far plane of rays."""
    num_levels: int = 16
    """Number of levels of the hashmap for the base MLP."""
    min_res: int = 16
    """Minimum resolution of the hashmap for the base MLP."""
    max_res: int = 8192
    """Maximum resolution of the hashmap for the base MLP."""
    log2_hashmap_size: int = 21
    """Size of the hashmap for the base MLP."""
    features_per_level: int = 2
    """Number of features per level of the hashmap for the base MLP."""
    num_layers: int = 2
    """Number of hidden layers for the base MLP."""
    hidden_dim: int = 256
    """Dimension of hidden layers for the base MLP."""
    num_layers_colour: int = 3
    """Number of hidden layers for colour MLP."""
    hidden_dim_colour: int = 256
    """Dimension of hidden layers for colour MLP."""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Implementation of the MLPs (tcnn or torch)."""
    object_density_bias: float = 0.0
    """Bias for object density."""
    num_proposal_samples_per_ray: Tuple[int, ...] = (512, 512)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Whether to use the same proposal network."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 5,
                "max_res": 512,
                "use_linear": False,
            },
            {
                "hidden_dim": 16,
                "log2_hashmap_size": 17,
                "num_levels": 7,
                "max_res": 2048,
                "use_linear": False,
            },
        ]
    )
    """Arguments for the proposal density fields."""
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    """Initial sampler for the proposal network."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing (this gives an exploration at the \
        beginning of training)."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 5000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    disable_scene_contraction: bool = False
    """Whether to disable scene contraction or not."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to \
        the camera."""
    rgb_loss_type: Literal["Bayer_mask", "Normal", "Robust", "Reweight" ] = "Reweight"
    """RGB Loss function"""
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    """Camera optimizer config"""
    use_depth_renderer: bool = True
    """Whether to use depth render for removing dynamic objects in the scene."""
    start_depth_render: int = 5000
    """Number of steps after which to start using depth renderer"""
    appearance_embed_dim: int = 32
    """Dimension of the appearance embedding."""
    use_appearance_embedding: bool = True
    """Whether to use an appearance embedding."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    robust_losses_delay: int = 15000
    """Number of epochs after which to start using the robust losses model"""
    average_init_density: int = 1.0
    """Average intial density"""
    distortion_loss_mult: float = 0.002
    """Distortion loss scalar"""
    pose_refinement: bool = False
    """Toggle pose refinement"""
    depth_renderer_distribution_base: float = 0.0
    """Depth based density reweighting distribution base value"""

class UnderWaterModel(Model):
    """Subsea model

    Args:
        config: Subsea Model configuration to instantiate the model with.
    """

    config: UnderWaterModelConfig

    def populate_modules(self):
        """Setup the fields and modules."""
        super().populate_modules()

        # Scene contraction
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0
        # Initialize SeaThru field
        self.field = UnderWaterField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            num_levels=self.config.num_levels,
            min_res=self.config.min_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            num_layers_colour=self.config.num_layers_colour,
            hidden_dim_colour=self.config.hidden_dim_colour,
            spatial_distortion=scene_contraction,
            implementation=self.config.implementation,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density
        )

        if self.config.pose_refinement:
            self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(num_cameras=self.num_train_data, device="cpu")

        # Initialize proposal network(s) (this code snippet is taken from from nerfacto)
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert (len(self.config.proposal_net_args_list) == 1), "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    average_init_density=self.config.average_init_density,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Initial sampler
        initial_sampler = None  # None is for piecewise as default
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        # Proposal sampler
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # Renderers
        self.renderer_rgb = UnderWaterRGBRenderer(
            use_depth_renderer=self.config.use_depth_renderer,
            start_depth_render=self.config.start_depth_render,
            training=self.training,
            base_value=self.config.depth_renderer_distribution_base
        )

        self.renderer_depth = UnderWaterDepthRenderer(far_plane=self.config.far_plane, method="median")
        self.renderer_accumulation = AccumulationRenderer()

        # Losses
        self.rgb_loss = MSELoss()
        self.step = 0

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # Step member variable to keep track of the training step
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.pose_refinement:
            self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def step_cb(self, step) -> None:
        self.step = step

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:  # type: ignore
        # Pose tweaks
        if self.training and self.config.pose_refinement:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)

        ray_samples: RaySamples

        # Get output from proposal network(s)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples)
        advisor_depth = self.renderer_depth(weights=field_outputs[FieldHeadNames.DENSITY], ray_samples=ray_samples)
        average_depth = torch.mean(advisor_depth)
        gradient_scaling_depth_threshold = 0.02 
        if self.config.use_gradient_scaling and average_depth < gradient_scaling_depth_threshold:
            field_outputs = scale_gradients_by_distance_squared(field_outputs=field_outputs, ray_samples=ray_samples)

        field_outputs[FieldHeadNames.DENSITY] = torch.nan_to_num(
            field_outputs[FieldHeadNames.DENSITY], nan=1e-3
        )
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
   
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        rgb, reweightings = self.renderer_rgb(
            object_rgb=field_outputs[FieldHeadNames.RGB],
            densities=field_outputs[FieldHeadNames.DENSITY],
            weights=weights,
            ray_samples=ray_samples,
            depth=depth,
            step=self.step
        )
        weights_list.append(reweightings)

        # Render depth and accumulation
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
            "weights": weights,
            "reweighting": reweighting_rgb_loss_rectifier(weights, reweightings)
        }

        # Add outputs from proposal network(s) to outputs if training for proposal loss
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

        # Add proposed depth to outputs
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_rgb = batch["image"].to(self.device)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])
        
        if self.config.pose_refinement:
            self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = outputs['rgb'], image

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            gt_image=image
        )

        # RGB Loss
        if self.config.rgb_loss_type == "Robust" and self.step > self.config.robust_losses_delay:
            loss_dict["rgb_loss"] = robust_loss(gt=gt_rgb, pred=pred_rgb, patch_size=16)
        elif self.config.rgb_loss_type == "Reweight":
            scaling = torch.mean(outputs["reweighting"])
            print(scaling)
            loss_dict["rgb_loss"] = recon_loss(gt=gt_rgb, pred=pred_rgb)
        else:
            loss_dict["rgb_loss"] = recon_loss(gt=gt_rgb, pred=pred_rgb) 

        if self.training:
            # Proposal loss
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.pose_refinement:
                self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)

        # Accumulation and depth maps
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(outputs["depth"])
        reweighting = colormaps.apply_depth_colormap(outputs["reweighting"])

        combined_rgb = torch.cat([gt_rgb, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Log the images
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "reweighting": reweighting
        }


        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # Compute metrics
        psnr = self.psnr(gt_rgb, rgb)
        ssim = self.ssim(gt_rgb, rgb)
        lpips = self.lpips(gt_rgb, rgb)

        # Log the metrics (as scalars)
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # Log the proposal depth maps
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(outputs[key])
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
