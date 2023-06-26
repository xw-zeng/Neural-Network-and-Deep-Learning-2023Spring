# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import layers

class ModulatedSiren(nn.Layer):
    """SIREN model with FiLM modulations as in pi-GAN."""
    def __init__(self,
        width: int = 256,
        depth: int = 5,
        in_channels: int = 2,
        out_channels: int = 3,
        w0: float = 1.,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        use_meta_sgd: bool = False,
        meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
        meta_sgd_clip_range: Tuple[float, float] = (-5., 5.)):
        """Constructor.

        Args:
        width (int): Width of each hidden layer in MLP.
        depth (int): Number of layers in MLP.
        in_channels (int): Number of input channels, i.e, coords for pixel(2D) and voxel(3D)
        out_channels (int): Number of output channels.
        w0 (float): Scale factor in sine activation in first layer.
        modulate_scale: If True, modulates scales.
        modulate_shift: If True, modulates shifts.
        use_meta_sgd: Whether to use meta-SGD.
        meta_sgd_init_range: Range from which initial meta_sgd learning rates will
            be uniformly sampled.
        meta_sgd_clip_range: Range at which to clip learning rates.
        name: name.
        """
        super().__init__()
        self.out_channels = out_channels
        self.use_meta_sgd = use_meta_sgd
        self.meta_sgd_init_range = meta_sgd_init_range
        self.meta_sgd_clip_range = meta_sgd_clip_range

        # Initialize meta-SGD learning rates
        # if self.use_meta_sgd:
        #   # Compute total number of modulations in network
        #   self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        #   self.num_modulations = width * (depth - 1) * self.modulations_per_unit
        #   self.meta_sgd_lrs = MetaSGDLrs(self.num_modulations,
        #                                  self.meta_sgd_init_range,
        #                                  self.meta_sgd_clip_range)

        siren_layers = []
        for i in range(depth):
            siren_layers.append(
                layers.ModulatedSirenLayer(
                    f_in=in_channels if i == 0 else width,
                    f_out=width if i != depth - 1 else self.out_channels,
                    is_first= i == 0,
                    is_last= i == depth - 1,
                    w0=w0,
                    modulate_scale=modulate_scale,
                    modulate_shift=modulate_shift)
            )
        self.siren_layers = nn.Sequential(*siren_layers)

    def __call__(self, coords: paddle.Tensor) -> paddle.Tensor:
        """Evaluates model at a batch of coordinates.

        Args:
            coords (Array): Array of coordinates. Should have shape (height, width, 2)
            for images and (depth/time, height, width, 3) for 3D shapes/videos.

        Returns:
            Output features at coords.
        """
        # Flatten coordinates
        x = paddle.reshape(coords, (-1, coords.shape[-1]))
        out = self.siren_layers(x)
        # Unflatten output. E.g. for images this corresponds to
        # (num_pixels, out_channels) -> (height, width, out_channels)
        return paddle.reshape(out, list(coords.shape[:-1]) + [self.out_channels])


class LatentModulatedSiren(nn.Layer):
    """SIREN model with FiLM modulations generated from a latent vector."""
    def __init__(self,
        batch_size: int = 1,
        width: int = 256,
        depth: int = 5,
        in_channels: int = 2,
        out_channels: int = 3,
        latent_dim: int = 64,
        layer_sizes: Tuple[int, ...] = (256, 512),
        w0: float = 1.,
        modulate_scale: bool = True,
        modulate_shift: bool = True,
        latent_init_scale: float = 0.01,
        use_meta_sgd: bool = False,
        meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
        meta_sgd_clip_range: Tuple[float, float] = (-5., 5.)):
        """Constructor.

        Args:
            width (int): Width of each hidden layer in MLP.
            depth (int): Number of layers in MLP.
            out_channels (int): Number of output channels.
            latent_dim: Dimension of latent vector (input of LatentToModulation
            network).
            layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
            w0 (float): Scale factor in sine activation in first layer.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            latent_init_scale: Scale at which to randomly initialize latent vector.
            use_meta_sgd: Whether to use meta-SGD.
            meta_sgd_init_range: Range from which initial meta_sgd learning rates will
            be uniformly sampled.
            meta_sgd_clip_range: Range at which to clip learning rates.
        """
        super().__init__()
        self.width = width
        self.depth = depth
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.latent_init_scale = latent_init_scale
        self.use_meta_sgd = use_meta_sgd
        self.meta_sgd_init_range = meta_sgd_init_range
        self.meta_sgd_clip_range = meta_sgd_clip_range

        # Initialize meta-SGD learning rates
        if self.use_meta_sgd:
            self.meta_sgd_lrs = layers.MetaSGDLrs(self.latent_dim,
                                            self.meta_sgd_init_range,
                                            self.meta_sgd_clip_range)

        # Initialize latent vector and map from latents to modulations
        self.latent = layers.LatentVector(latent_dim, batch_size, latent_init_scale)
        self.latent_to_modulation = layers.LatentToModulation(
            latent_dim=latent_dim,
            layer_sizes=layer_sizes,
            width=width,
            num_modulation_layers=depth - 1,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift)

        siren_layers = []
        for i in range(depth):
            siren_layers.append(
                layers.ModulatedSirenLayer(
                    f_in=in_channels if i == 0 else width,
                    f_out=width if i != depth - 1 else self.out_channels,
                    is_first= i == 0,
                    is_last= i == depth - 1,
                    w0=w0,
                    modulate_scale=False,
                    modulate_shift=False,
                    apply_activation=False)
            )
        self.siren_layers = nn.LayerList(siren_layers)
        self.sine = layers.Sine(self.w0)

    def modulate(self, x: paddle.Tensor, modulations: Dict[str, paddle.Tensor]) -> paddle.Tensor:
        """Modulates input according to modulations.

        Args:
            x: Hidden features of MLP.
            modulations: Dict with keys 'scale' and 'shift' (or only one of them)
            containing modulations.

        Returns:
            Modulated vector.
        """
        if 'scale' in modulations:
            x = modulations['scale'].unsqueeze(1) * x
        if 'shift' in modulations:
            x = x + modulations['shift'].unsqueeze(1)
        return x

    def __call__(self, coords, latent_vector=None):
        """Evaluates model at a batch of coordinates.

        Args:
            coords (Array): Array of coordinates. Should have shape (height, width, 2)
            for images and (depth/time, height, width, 3) for 3D shapes/videos.

        Returns:
            Output features at coords.
        """
        # Compute modulations based on latent vector
        latent_vector = self.latent() if latent_vector is None else latent_vector
        modulations = self.latent_to_modulation(latent_vector)

        # Flatten coordinates
        x = paddle.reshape(coords, (coords.shape[0], -1, coords.shape[-1]))

        # Initial layer (note all modulations are set to False here, since we
        # directly apply modulations from latent_to_modulations output).
        x = self.siren_layers[0](x)
        x = self.modulate(x, modulations[0])
        x = self.sine(x)

        # Hidden layers
        for i in range(1, self.depth - 1):
            x = self.siren_layers[i](x)
            x = self.modulate(x, modulations[i])
            x = self.sine(x)

        # Final layer
        out = self.siren_layers[-1](x)

        # Unflatten output
        return paddle.reshape(out, list(coords.shape[:-1]) + [self.out_channels])
