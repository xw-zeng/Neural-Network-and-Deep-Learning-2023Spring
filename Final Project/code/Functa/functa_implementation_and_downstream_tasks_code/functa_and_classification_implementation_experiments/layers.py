from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Sine(nn.Layer):
  """Applies a scaled sine transform to input: out = sin(w0 * in)."""

  def __init__(self, w0: float = 1.):
    """Constructor.

    Args:
      w0 (float): Scale factor in sine activation (omega_0 factor from SIREN).
    """
    super().__init__()
    self.w0 = w0

  def __call__(self, x: paddle.Tensor) -> paddle.Tensor:
    return paddle.sin(self.w0 * x)


class FiLM(nn.Layer):
    """Applies a FiLM modulation: out = scale * in + shift.

    Notes:
    We currently initialize FiLM layers as the identity. However, this may not
    be optimal. In pi-GAN for example they initialize the layer with a random
    normal.
    """
    def __init__(self,
                f_in: int,
                modulate_scale: bool = True,
                modulate_shift: bool = True):
        """Constructor.

        Args:
        f_in: Number of input features.
        modulate_scale: If True, modulates scales.
        modulate_shift: If True, modulates shifts.
        """
        super().__init__()
        # Must modulate at least one of shift and scale
        assert modulate_scale or modulate_shift
        # Initialize FiLM layers as identity
        self.scale = 1.
        self.shift = 0.
        if modulate_scale:
            self.scale = self.create_parameter(shape=[f_in], default_initializer=nn.initializer.Constant(value=1.))
        if modulate_shift:
            self.shift = self.create_parameter(shape=[f_in], default_initializer=nn.initializer.Constant(value=0.))

    def __call__(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.scale * x + self.shift


class ModulatedSirenLayer(nn.Layer):
    """Applies a linear layer followed by a modulation and sine activation."""
    def __init__(self,
                f_in: int,
                f_out: int,
                w0: float = 1.,
                is_first: bool = False,
                is_last: bool = False,
                modulate_scale: bool = True,
                modulate_shift: bool = True,
                apply_activation: bool = True):
        """Constructor.

        Args:
        f_in (int): Number of input features.
        f_out (int): Number of output features.
        w0 (float): Scale factor in sine activation.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model.
        modulate_scale: If True, modulates scales.
        modulate_shift: If True, modulates shifts.
        apply_activation: If True, applies sine activation.
        """
        super().__init__()
        self.is_last = is_last
        # Follow initialization scheme from SIREN
        init_range = 1 / f_in if is_first else np.sqrt(6 / f_in) / w0

        self.fc = nn.Linear(f_in, f_out, 
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(-init_range, init_range)
            )
        )

        if not is_last and (modulate_scale or modulate_shift):
            self.film = FiLM(f_out, modulate_scale, modulate_shift)
        else:
            self.film = None

        self.act = Sine(w0) if apply_activation else None

    def __call__(self, x: paddle.Tensor) -> paddle.Tensor:
        # Shape (n, f_in) -> (n, f_out)
        x = self.fc(x)
        # Apply non-linearities
        if self.is_last:
            # We assume target data (e.g. RGB values of pixels) lies in [0, 1]. To
            # learn zero-centered features we therefore shift output by .5
            return x + .5
        else:
            # Optionally apply modulation
            if self.film is not None:
                x = self.film(x)
            # Optionally apply activation
            if self.act is not None:
                x = self.act(x)
            return x


class LatentVector(nn.Layer):
    """Module that holds a latent vector.

    Notes:
    This module does not apply any transformation but simply stores a latent
    vector. This is to make sure that all data necessary to represent an image
    (or a NeRF scene or a video) is present in the model params. This also makes
    it easier to use the partition_params function.
    """
    def __init__(self, latent_dim: int, batch_size: int = 1, latent_init_scale: float = 0.0):
        """Constructor.

        Args:
        latent_dim: Dimension of latent vector.
        latent_init_scale: Scale at which to randomly initialize latent vector.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_init_scale = latent_init_scale
        # Initialize latent vector
        self.latent_vector = self.create_parameter([batch_size, latent_dim], 
            default_initializer=nn.initializer.Uniform(-latent_init_scale, latent_init_scale))

    def __call__(self):
        return self.latent_vector


class LatentToModulation(nn.Layer):
    """Function mapping latent vector to a set of modulations."""
    def __init__(self,
               latent_dim: int,
               layer_sizes: Tuple[int, ...],
               width: int,
               num_modulation_layers: int,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               activation: str = "relu"):
        """Constructor.
        Args:
            latent_dim: Dimension of latent vector (input of LatentToModulation
            network).
            layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
            width: Width of each hidden layer in MLP of function rep.
            num_modulation_layers: Number of layers in MLP that contain modulations.
            modulate_scale: If True, returns scale modulations.
            modulate_shift: If True, returns shift modulations.
            activation: Activation function to use in MLP.
        """
        super().__init__()
        # Must modulate at least one of shift and scale
        assert modulate_scale or modulate_shift

        self.latent_dim = latent_dim
        self.layer_sizes = list(layer_sizes)  # counteract XM that converts to list
        self.width = width
        self.num_modulation_layers = num_modulation_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

        # MLP outputs all modulations. We apply modulations on every hidden unit
        # (i.e on width number of units) at every modulation layer.
        # At each of these we apply either a scale or a shift or both,
        # hence total output size is given by following formula
        self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        self.modulations_per_layer = width * self.modulations_per_unit
        self.output_size = num_modulation_layers * self.modulations_per_layer

        # self.forward = hk.nets.MLP(
        #     self.layer_sizes + (self.output_size,), activation=activation)
        linear_params = [latent_dim] + self.layer_sizes + [self.output_size]

        linears = []
        for k in range(len(linear_params) - 1):
            linears.append(nn.Linear(linear_params[k], linear_params[k + 1]))
            if k != len(linear_params) - 2:
                if activation == "relu":
                    linears.append(nn.ReLU())
        self.linear = nn.Sequential(*linears)

    def __call__(self, latent_vector):
        modulations = self.linear(latent_vector)

        # Partition modulations into scales and shifts at every layer
        outputs = {}
        for i in range(self.num_modulation_layers):
            single_layer_modulations = {}
            # Note that we add 1 to scales so that outputs of MLP will be centered
            # (since scale = 1 corresponds to identity function)
            if self.modulate_scale and self.modulate_shift:
                start = 2 * self.width * i
                single_layer_modulations['scale'] = modulations[:, start:start +
                                                                self.width] + 1
                single_layer_modulations['shift'] = modulations[:, start +
                                                                self.width:start +
                                                                2 * self.width]
            elif self.modulate_scale:
                start = self.width * i
                single_layer_modulations['scale'] = modulations[:, start:start +
                                                                self.width] + 1
            elif self.modulate_shift:
                start = self.width * i
                single_layer_modulations['shift'] = modulations[:, start:start +
                                                                self.width]
            outputs[i] = single_layer_modulations
        return outputs


class MetaSGDLrs(nn.Layer):
    """Module storing learning rates for meta-SGD.

    Notes:
    This module does not apply any transformation but simply stores the learning
    rates. Since we also learn the learning rates we treat them the same as
    model params.
    """

    def __init__(self,
               num_lrs: int,
               lrs_init_range: Tuple[float, float] = (0.005, 0.1),
               lrs_clip_range: Tuple[float, float] = (-5., 5.)):
        """Constructor.

        Args:
            num_lrs: Number of learning rates to learn.
            lrs_init_range: Range from which initial learning rates will be
            uniformly sampled.
            lrs_clip_range: Range at which to clip learning rates. Default value will
            effectively avoid any clipping, but typically learning rates should
            be positive and small.
        """
        super().__init__()
        self.lrs_init_range = lrs_init_range
        self.lrs_clip_range = lrs_clip_range
        # Initialize learning rates
        self.meta_sgd_lrs = self.create_parameter([num_lrs], 
            default_initializer=nn.initializer.Uniform(*lrs_init_range))

    def __call__(self):
        # Clip learning rate values
        return paddle.clip(self.meta_sgd_lrs, *self.lrs_clip_range)