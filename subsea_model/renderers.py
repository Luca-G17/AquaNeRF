from typing import Literal, Union, Tuple

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples

class UnderWaterRGBRenderer(nn.Module):
    """Volumetric RGB rendering of an unnderwater scene.

    Args:
    """

    def __init__(self, use_depth_renderer: bool = True, start_depth_render: int = 3500, training: bool = True, base_value: float = 0.0) -> None:
        super().__init__()
        self.use_depth_renderer = use_depth_renderer
        self.start_depth_render = start_depth_render
        self.training = training
        self.base_value = base_value

    def combine_rgb(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
        depth: Float[Tensor, "*bs 1"],
        step: int
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            densities: Object densities.
            weights: Object weights.
            ray_samples: Set of ray samples.

        Returns:
            Rendered pixel colour.
        """
        std  = 0.10
        assume_medium_threshold = 0.2
        s = ray_samples.frustums.starts

        comp_object_rgb = 0
        depth_weighting = weights
        if (self.use_depth_renderer and step > self.start_depth_render) or (self.use_depth_renderer and not(self.training)):
            depth_weighting = torch.clip(self.base_value + torch.exp(-torch.square(s - depth.unsqueeze(1)) / (2 * (std ** 2))), 0, 1)

            # Only apply depth renderer if the accumulated absorption passes some threshold - i.e. the ray hits an object in the foreground
            accumulated_weight = torch.sum(weights, dim=-2)
            #assume_medium = depth > 100
            #depth_weighting[assume_medium.squeeze()] = 1

            #print(depth)
            comp_object_rgb = torch.sum(weights * depth_weighting * rgb, dim=-2)
        else:
            comp_object_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        background_color = rgb[..., -1, :]
        comp_rgb = comp_object_rgb + background_color.expand(comp_object_rgb.shape).to(comp_object_rgb.device) * (1.0 - accumulated_weight)

        return comp_rgb, weights * depth_weighting

    def blend_background(
        self,
        image: Tensor,
    ) -> Float[Tensor, "*bs 3"]:
        """Blends the background colour into the image if image is RGBA.
        Otherwise no blending is performed (we assume opacity of 1).

        Args:
            image: RGB/RGBA per pixel.
            opacity: Alpha opacity per pixel.
        Returns:
            Blended RGB.
        """
        if image.size(-1) < 4:
            return image

        rgb, opacity = image[..., :3], image[..., 3:]
        background_color =  torch.tensor([0.0, 0.0, 0.0]).expand(rgb.shape).to(rgb.device)
        return rgb * opacity + background_color.to(rgb.device) * (1 - opacity)

    def blend_background_for_loss_computation(
        self,
        pred_image: Tensor,
        gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Blends a background color into the ground truth and predicted image for
        loss computation.

        Args:
            gt_image: The ground truth image.
            pred_image: The predicted RGB values (without background blending).
            pred_accumulation: The predicted opacity/ accumulation.
        Returns:
            A tuple of the predicted and ground truth RGB values.
        """
        gt_image = self.blend_background(gt_image)
        return pred_image, gt_image

    def forward(
        self,
        object_rgb: Float[Tensor, "*bs num_samples 3"],
        densities: Float[Tensor, "*bs num_samples 1"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_samples: RaySamples,
        depth: Float[Tensor, "*bs 1"],
        step: int
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Render pixel colour along rays using volumetric rendering.

        Args:
            object_rgb: RGB values of object.
            densities: Object densities.
            weights: Object weights.
            ray_samples: Set of ray samples.
            depth: Max absorption depth
            step: Training step number

        Returns:
            Rendered pixel colour.
        """
        if not self.training:
            object_rgb = torch.nan_to_num(object_rgb)

            rgb = self.combine_rgb(
                object_rgb,
                densities,
                weights,
                ray_samples=ray_samples,
                depth=depth,
                step=step
            )

            if isinstance(rgb, torch.Tensor):
                torch.clamp_(rgb, min=0.0, max=1.0)

            return rgb

        else:
            rgb = self.combine_rgb(
                object_rgb,
                densities,
                weights,
                ray_samples=ray_samples,
                depth=depth,
                step=step
            )
            return rgb


class UnderWaterDepthRenderer(nn.Module):
    def __init__(
        self, far_plane: float, method: Literal["median", "expected"] = "median"
    ) -> None:
        super().__init__()
        self.far_plane = far_plane
        self.method = method

    def forward(
        self,
        weights: Float[Tensor, "*batch num_samples 1"],
        ray_samples: RaySamples,
    ) -> Float[Tensor, "*batch 1"]:

        steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        if self.method == "expected":
            eps = 1e-10
            depth = torch.sum(weights * steps, dim=-2) / (torch.sum(weights, -2) + eps)
            depth = torch.clip(depth, steps.min(), steps.max())
            return depth

        if self.method == "median":
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
            split = (torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.55)  # [..., 1]
            median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
            median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
            median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]
            return median_depth

        raise NotImplementedError(f"Method {self.method} not implemented")
