from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import torch
from nerfstudio.engine.optimizers import OptimizerConfig

@dataclass 
class RMSPropOptimizerConfig(OptimizerConfig):
    _target: Type = torch.optim.RMSprop
    """RMSProp Parameters: """
    weight_decay: float = 0.0
    momentum: float = 0.0
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        kwargs.pop("eps")
        return self._target(params, **kwargs)

@dataclass 
class SDGOptimizerConfig(OptimizerConfig):
    _target: Type = torch.optim.SGD
    """SGD Parameters: """
    weight_decay: float = 0.0
    momentum: float = 0.0
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        kwargs.pop("eps")
        return self._target(params, **kwargs)