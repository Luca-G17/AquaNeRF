import torch
from torch import Tensor
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from typing import Dict, Tuple, cast
from nerfstudio.model_components.losses import _GradientScaler

def acc_loss(
    transmittance_object: Float[Tensor, "*bs num_samples 1"], beta: float
) -> torch.Tensor:
    P = torch.exp(-torch.abs(transmittance_object) / 0.1) + beta * torch.exp(
        -torch.abs(1 - transmittance_object) / 0.1
    )
    loss = -torch.log(P)
    return loss.mean()


def recon_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute the reconstruction loss.

    Args:
        gt: Ground truth.
        pred: RGB prediction.

    Returns:
        Reconstruction loss.
    """
    loss = torch.mean(torch.square((pred - gt) / (pred.detach() + 1e-3)))
    return loss

def scale_gradients_by_distance_squared(field_outputs: Dict[FieldHeadNames, torch.Tensor], ray_samples: RaySamples) -> Dict[FieldHeadNames, torch.Tensor]:
    out = {}
    ray_dist = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    scaling = torch.square(ray_dist).clamp(0, 1)
    for key, value in field_outputs.items():
        out[key], _ = cast(Tuple[Tensor, Tensor], _GradientScaler.apply(value, scaling))
    return out

def robust_loss(gt: torch.Tensor, pred: torch.Tensor, patch_size: int) -> torch.Tensor:
    pred_patches = pred.view(-1, 1, patch_size, patch_size, 3)
    gt_patches = gt.view(-1, 1, patch_size, patch_size, 3)
    device = pred.device
    batch_size = pred_patches.shape[0]
    residuals = torch.mean((pred_patches - gt_patches)**2, dim=-1)
    with torch.no_grad():
        med_residual = torch.quantile(residuals, .9)
        weight = (residuals <= med_residual).float()
        weight = torch.nn.functional.pad(weight, (1,1,1,1), mode='replicate')
        blurred_w = (torch.nn.functional.conv2d(weight, (1/9.) * torch.ones((1,1,3,3), device=device), padding='valid') >= 0.5).float()
        expected_w = blurred_w.view(batch_size, -1).mean(1)
        weight_r8 = (expected_w >= 0.6).float()
    return torch.mean(residuals.squeeze() * weight_r8[:,None,None])

def reweighting_rgb_loss_rectifier(weights, reweighted):
    reweight_diff = torch.mean(torch.square(weights - reweighted), 1)
    return reweight_diff