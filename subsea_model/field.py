import torch
from torch import nn
from torch import Tensor

from typing import Dict, Literal, Optional, Tuple
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP, MLPWithHashEncoding
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import HashEncoding, SHEncoding, NeRFEncoding
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.field_components.embedding import Embedding

try:
    import tinycudann as tcnn  # noqa
except ImportError:
    print("tinycudann is not installed! Please install it for faster training.")


class UnderWaterField(Field):
    """Field:

    Args:
        aabb: parameters of scene aabb bounds
        num_levels: number of levels of the hashmap for the object base MLP
        min_res: minimum resolution of the hashmap for the object base MLP
        max_res: maximum resolution of the hashmap for the object base MLP
        log2_hashmap_size: size of the hashmap for the object base MLP
        features_per_level: number of features per level of the hashmap for the object
                            base MLP
        num_layers: number of hidden layers for the object base MLP
        hidden_dim: dimension of hidden layers for the object base MLP
        bottleneck_dim: bottleneck dimension between object base MLP and object head MLP
        num_layers_colour: number of hidden layers for colour MLP
        hidden_dim_colour: dimension of hidden layers for colour MLP
        spatial_distortion: spatial distortion to apply to the scene
        implementation: implementation of the base mlp (tcnn or torch)
        use_viewing_dir_obj_rgb: whether to use viewing direction in object rgb MLP
        object_density_bias: bias for object density
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        num_layers: int = 2,
        hidden_dim: int = 64,
        num_layers_colour: int = 3,
        hidden_dim_colour: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        object_density_bias: float = 0.0,
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,
        geo_feat_dim: int = 15,
        average_init_density: float = 1.0
    ) -> None:
        super().__init__()

        # Register buffers
        self.register_buffer("aabb", aabb)
        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.geo_feat_dim = geo_feat_dim
        self.num_images = num_images
        self.spatial_distortion = spatial_distortion
        self.object_density_bias = object_density_bias
        self.appearance_embedding_dim = appearance_embedding_dim
        self.average_init_density = average_init_density
        if self.appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.embedding_appearance = None
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.direction_encoding = SHEncoding(levels=4, implementation=implementation)
        self.position_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2 - 1, implementation=implementation)

        self.mlp_base = MLPWithHashEncoding(
            num_levels=num_levels,
            min_res=min_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation
        )

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_colour,
            layer_width=hidden_dim_colour,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Compute output of base MLP.

        Args:
            ray_samples: RaySamples object containing the ray samples.

        Returns:
            Tuple containing the object density and the bottleneck vector.
        """
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        # Make sure the tcnn gets inputs between 0 and 1
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions

        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True

        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation
        # From nerfacto: "Rectifying the density with an exponential is much more stable
        # than a ReLU or softplus, because it enables high post-activation (float32)
        # density outputs from smaller internal (float16) parameters."
        density = self.average_init_density * trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        
        camera_indices = ray_samples.camera_indices.squeeze()

        # Encode directions
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        directions_encoded = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones((*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros((*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device)

        h = torch.cat([directions_encoded, density_embedding.view(-1, self.geo_feat_dim)] + 
                      ([embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []), dim=-1)
    

        # Object colour MLP forward pass
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
