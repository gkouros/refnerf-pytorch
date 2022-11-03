# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NeRF and its MLPs, with helper functions for construction and rendering."""

import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Tuple

from flax import linen as nn
import gin
import torch
from torch import nn
from internal import configs
from internal import coord
from internal import geopoly
from internal import image
from internal import math
from internal import ref_utils
from internal import render
from internal import stepfun
from internal import utils

gin.config.external_configurable(math.safe_exp, module='math')
gin.config.external_configurable(coord.contract, module='coord')


@gin.configurable
class Model(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""
    config: Any = None  # A Config class, must be set upon construction.
    # The number of samples for each proposal level.
    num_prop_samples: int = 64
    num_nerf_samples: int = 32  # The number of samples the final nerf level.
    # The number of sampling levels (3==2 proposals, 1 nerf).
    num_levels: int = 3
    # The range of background colors.
    bg_intensity_range: Tuple[float] = (1., 1.)
    use_viewdirs: bool = True  # If True, use view directions as input.
    raydist_fn: Callable[..., Any] = None  # The curve used for ray dists.
    ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
    disable_integration: bool = False  # If True, use PE instead of IPE.
    # If True, jitter whole rays instead of samples.
    single_jitter: bool = True
    single_mlp: bool = False  # Use the NerfMLP for all rounds of sampling.
    # Dirichlet/alpha "padding" on the histogram.
    resample_padding: float = 0.0
    opaque_background: bool = False  # If true, make the background opaque.
    # initial values for near and far bounds of the rays
    init_s_near: float = 0.
    init_s_far: float = 1.

    def setup(self):
        # Construct MLPs. WARNING: Construction order may matter, if MLP weights are
        # being regularized.
        self.nerf_mlp = NerfMLP()
        self.prop_mlp = nerf_mlp if self.single_mlp else PropMLP()

    def __call__(
        self,
        rays,
        train_frac,
        compute_extras,
    ):
        """The mip-NeRF Model.

        Args:
          rays: util.Rays, a pytree of ray origins, directions, and viewdirs.
          train_frac: float in [0, 1], what fraction of training is complete.
          compute_extras: bool, if True, compute extra quantities besides color.

        Returns:
          ret: list, [*(rgb, distance, acc)]
        """

        # Define the mapping from normalized to metric ray distance.
        _, s_to_t = coord.construct_ray_warps(
            self.raydist_fn, rays.near, rays.far)

        # Initialize the range of (normalized) distances for each ray to [0, 1],
        # and assign that single interval a weight of 1. These distances and weights
        # will be repeatedly updated as we proceed through sampling levels.
        sdist = torch.concatenate([
            torch.full_like(rays.near, self.init_s_near),
            torch.full_like(rays.far, self.init_s_far)
        ],
            axis=-1)
        weights = torch.ones_like(rays.near)

        ray_history = []
        renderings = []
        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            # A slightly more stable way to compute weights**anneal. If the distance
            # between adjacent intervals is zero then its weight is fixed to 0.
            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                torch.log(weights + self.resample_padding), -float('inf'))

            # Draw sampled intervals from each ray's current weights.
            sdist = stepfun.sample_intervals(
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(self.init_s_near, self.init_s_far),
                use_gpu_resampling=False,
                )

            # Optimization will usually go nonlinear if you propagate gradients
            # through sampling.
            sdist.detach()

            # Convert normalized distances to metric distances.
            tdist = s_to_t(sdist)

            # Cast our rays, by turning our distance intervals into Gaussians.
            gaussians = render.cast_rays(
                tdist,
                rays.origins,
                rays.directions,
                rays.radii,
                self.ray_shape,
                diag=False)

            if self.disable_integration:
                # Setting the covariance of our Gaussian samples to 0 disables the
                # "integrated" part of integrated positional encoding.
                gaussians = (gaussians[0], torch.zeros_like(gaussians[1]))

            # Push our Gaussians through one of our two MLPs.
            mlp = self.prop_mlp if is_prop else self.nerf_mlp
            ray_results = mlp(
                gaussians,
                viewdirs=rays.viewdirs if self.use_viewdirs else None,
                imageplane=rays.imageplane,
            )

            # Get the weights used by volumetric rendering (and our other losses).
            weights = render.compute_alpha_weights(
                ray_results['density'],
                tdist,
                rays.directions,
                opaque_background=self.opaque_background,
            )[0]

            # Define or sample the background color for each ray.
            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                # If the min and max of the range are equal, just take it.
                bg_rgbs = self.bg_intensity_range[0]
            else:
                # If rendering is deterministic, use the midpoint of the range.
                bg_rgbs = (
                    self.bg_intensity_range[0] + self.bg_intensity_range[1]) / 2

            # Render each ray.
            rendering = render.volumetric_rendering(
                ray_results['rgb'],
                weights,
                tdist,
                bg_rgbs,
                rays.far,
                compute_extras,
                extras={
                    k: v
                    for k, v in ray_results.items()
                    if k.startswith('normals') or k in ['roughness']
                })

            if compute_extras:
                # Collect some rays to visualize directly. By naming these quantities
                # with `ray_` they get treated differently downstream --- they're
                # treated as bags of rays, rather than image chunks.
                n = self.config.vis_num_rays
                rendering['ray_sdist'] = sdist.reshape(
                    [-1, sdist.shape[-1]])[:n, :]
                rendering['ray_weights'] = (
                    weights.reshape([-1, weights.shape[-1]])[:n, :])
                rgb = ray_results['rgb']
                rendering['ray_rgbs'] = (rgb.reshape(
                    (-1,) + rgb.shape[-2:]))[:n, :, :]

            renderings.append(rendering)
            ray_results['sdist'] = torch.copy(sdist)
            ray_results['weights'] = torch.copy(weights)
            ray_history.append(ray_results)

        if compute_extras:
            # Because the proposal network doesn't produce meaningful colors, for
            # easier visualization we replace their colors with the final average
            # color.
            weights = [r['ray_weights'] for r in renderings]
            rgbs = [r['ray_rgbs'] for r in renderings]
            final_rgb = torch.sum(rgbs[-1] * weights[-1][..., None], dim=-2)
            avg_rgbs = [
                torch.broadcast_to(final_rgb[:, None, :], r.shape) for r in rgbs[:-1]
            ]
            for i in range(len(avg_rgbs)):
                renderings[i]['ray_rgbs'] = avg_rgbs[i]

        return renderings, ray_history


def construct_model(rays, config):
    """Construct a mip-NeRF 360 model.

    Args:
      rays: an example of input Rays.
      config: A Config class.

    Returns:
      model: initialized nn.Module, a NeRF model with parameters.
      init_variables: flax.Module.state, initialized NeRF model parameters.
    """
    # Grab just 10 rays, to minimize memory overhead during construction.
    ray = torch.reshape(x, [-1, x.shape[-1]])[:10]
    model = Model(config=config)
    init_variables = model.init(
        rays=ray,
        train_frac=1.,
        compute_extras=False)
    return model, init_variables


class MLP(nn.Module):
    """A PosEnc MLP."""
    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 256  # The width of the first part of MLP.
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_viewdirs: int = 1  # The depth of the second part of ML.
    net_width_viewdirs: int = 128  # The width of the second part of MLP.
    net_activation: Callable[..., Any] = nn.relu  # The activation function.
    min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
    max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
    weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
    # Add a skip connection to the output of every N layers.
    skip_layer: int = 4
    skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_directional_enc: bool = False  # If True, use IDE to encode directions.
    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False
    # Roughness activation function.
    roughness_activation: Callable[..., Any] = nn.softplus
    roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
    # If True, predict diffuse & specular colors.
    use_diffuse_color: bool = False
    use_specular_tint: bool = False  # If True, predict tint.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
    # Std. deviation of noise added to bottleneck.
    bottleneck_noise: float = 0.0
    density_activation: Callable[..., Any] = nn.softplus  # Density activation.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    enable_pred_normals: bool = False  # If True compute predicted normals.
    disable_density_normals: bool = False  # If True don't compute normals.
    disable_rgb: bool = False  # If True don't output RGB.
    warp_fn: Callable[..., Any] = None
    basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
    # Tesselation count. 'octahedron' + 1 == eye(3).
    basis_subdivisions: int = 2

    def setup(self):
        # Make sure that normals are computed if reflection direction is used.
        if self.use_reflections and not (self.enable_pred_normals or
                                         not self.disable_density_normals):
            raise ValueError(
                'Normals must be computed for reflection directions.')

        # Precompute and store (the transpose of) the basis being used.
        self.pos_basis_t = torch.array(
            geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)).T

        # Precompute and define viewdir or refdir encoding function.
        if self.use_directional_enc:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
        else:

            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

            self.dir_enc_fn = dir_enc_fn

        ############################ Spatial MLP ###############################
        spatial_layers = [nn.LazyLinear(self.net_width)]
        for i in range(self.net_depth - 1):
            spatial_layers += [
               nn.Linear(self.net_width + input_ch, self.net_width)
               if i % self.skip_layer
               else nn.Linear(self.net_width, self.net_width)
            ]
        self.spatial_net = nn.ModuleList(spatial_layers)

        # raw density layer
        self.raw_density = nn.Linear(self.net_width, 1)

        # predicted normals
        self.normals_pred = nn.Linear(self.net_width, 3)

        # roughness layer
        if self.enable_pred_roughness:
            self.raw_roughness = nn.Linear(self.net_width, 1)

        # diffuse layer
        if self.use_diffuse_color:
            self.raw_rgb_diffuse = nn.Linear(self.net_width, self.num_rgb_channels)

        # tint layer
        if self.use_specular_tint:
            self.raw_tint = nn.Linear(self.net_width, 3)

        # bottleneck layer
        if self.bottleneck_width > 0:
            self.bottleneck = nn.Linear(self.net_width, self.bottleneck_width)

        ########################## Directional MLP #############################
        viewdir_layers = [nn.LazyLinear(self.net_width_viewdirs)]
        for i in range(self.net_depth_viewdirs-1):
            viewdir_layers += [
               nn.Linear(self.net_width_viewdirs, self.net_width_viewdirs)
               if i % self.skip_layer_dir and i > 0
               else nn.Linear(self.net_width_viewdirs, self.net_width_viewdirs)
            ]
        self.viewdir_mlp = nn.ModuleList(viewdir_layers)

        # rgb layer
        self.rgb = nn.LazyLinear(self.num_rgb_channels)


    def __call__(self,
                 gaussians,
                 viewdirs=None,
                 imageplane=None,
        ):
        """Evaluate the MLP.

        Args:
          gaussians: a tuple containing:                                           /
            - mean: [..., n, 3], coordinate means, and                             /
            - cov: [..., n, 3{, 3}], coordinate covariance matrices.
          viewdirs: torch.ndarray(float32), [..., 3], if not None, this variable will
            be part of the input to the second part of the MLP concatenated with the
            output vector of the first part of the MLP. If None, only the first part
            of the MLP will be used with input x. In the original paper, this
            variable is the view direction.
          imageplane: torch.ndarray(float32), [batch, 2], xy image plane coordinates
            for each ray in the batch. Useful for image plane operations such as a
            learned vignette mapping.

        Returns:
          rgb: torch.ndarray(float32), with a shape of [..., num_rgb_channels].
          density: torch.ndarray(float32), with a shape of [...].
          normals_pred: torch.ndarray(float32), with a shape of [..., 3], or None.
          roughness: torch.ndarray(float32), with a shape of [..., 1], or None.
        """
        # get inputs in the form of means and variances representation the ray segments
        means, covs = gaussians
        # lift means and vars of position input
        lifted_means, lifted_vars = (
            coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
        # apply integrated position encoding to position input
        x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                     self.min_deg_point, self.max_deg_point)
        inputs = x
        # Evaluate network to produce the output density.
        inputs = x
        for i, layer in enumerate(self.spatial_net):
            x = layer(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.concatenate([x, inputs], dim=-1)
        raw_density = self.raw_density(x)[..., 0]

        # Add noise to regularize the density predictions if needed.
        if self.density_noise > 0:
            raw_density += self.density_noise * torch.normal(0, 1, raw_density.shape)

        # Apply bias and activation to raw density
        density = self.density_activation(raw_density + self.density_bias)

        # predict normals
        grad_pred = self.normals_pred(x)
        # Normalize negative predicted gradients to get predicted normal vectors.
        normals_pred = -ref_utils.l2_normalize(grad_pred)

        roughness = 0
        if self.disable_rgb:
            rgb = torch.zeros_like(means)
        else:
            if viewdirs is not None:
                # Predict diffuse color.
                if self.use_diffuse_color:
                    raw_rgb_diffuse = self.raw_rgb_diffuse(x)

                if self.use_specular_tint:
                    tint = nn.Sigmoid(self.raw_tint(x))

                if self.enable_pred_roughness:
                    roughness = self.roughness_activation(
                        self.raw_roughness(x) + self.roughness_bias)

                # Output of the first part of MLP.
                if self.bottleneck_width > 0:
                    bottleneck = self.bottleneck(x)

                    # Add bottleneck noise.
                    if self.bottleneck_noise > 0:
                        bottleneck += self.bottleneck_noise * torch.normal(
                            0, 1, bottleneck.shape)

                    x = [bottleneck]
                else:
                    x = []

                # Encode view (or reflection) directions.
                if self.use_reflections:
                    # Compute reflection directions. Note that we flip viewdirs before
                    # reflecting, because they point from the camera to the point,
                    # whereas ref_utils.reflect() assumes they point toward the camera.
                    # Returned refdirs then point from the point to the environment.
                    refdirs = ref_utils.reflect(
                        -viewdirs[..., None, :], normals_pred)
                    # Encode reflection directions.
                    dir_enc = self.dir_enc_fn(refdirs, roughness)
                else:
                    # Encode view directions.
                    dir_enc = self.dir_enc_fn(viewdirs, roughness)

                # broadcast directional encoding to bottleneck's dimensions
                dir_enc = torch.broadcast_to(
                    dir_enc[..., None, :],
                    bottleneck.shape[:-1] + (dir_enc.shape[-1],))

                # Append view (or reflection) direction encoding to bottleneck vector.
                x.append(dir_enc)

                # Append dot product between normal vectors and view directions.
                if self.use_n_dot_v:
                    dotprod = torch.sum(
                        normals_pred * viewdirs[..., None, :],
                        dim=-1, keepdims=True)
                    x.append(dotprod)

                # Concatenate bottleneck, directional encoding, and nv product
                x = torch.cat(x, dim=-1)

                # Output of the second part of MLP.
                x = self.viewdir_mlp(x)


            # If using diffuse/specular colors, then `rgb` is treated as linear
            # specular color. Otherwise it's treated as the color itself.
            rgb = self.rgb_activation(self.rgb_premultiplier * self.rgb(x) + self.rgb_bias)

            if self.use_diffuse_color:
                # Initialize linear diffuse color around 0.25, so that the combined
                # linear color is initialized around 0.5.
                diffuse_linear = nn.sigmoid(raw_rgb_diffuse - torch.log(3.0))
                if self.use_specular_tint:
                    specular_linear = tint * rgb
                else:
                    specular_linear = 0.5 * rgb

                # Combine specular and diffuse components and tone map to sRGB.
                rgb = torch.clip(
                    image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

            # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals_pred=normals_pred,
            roughness=roughness,
        )


@gin.configurable
class NerfMLP(MLP):
    pass


@gin.configurable
class PropMLP(MLP):
    pass


def render_image(render_fn: Callable[[torch.array, utils.Rays],
                                     Tuple[List[Mapping[Text, torch.ndarray]],
                                           List[Tuple[torch.ndarray, ...]]]],
                 rays: utils.Rays,
                 config: configs.Config,
                 verbose: bool = True) -> MutableMapping[Text, Any]:
    """Render all the pixels of an image (in test mode).

    Args:
      render_fn: function, jit-ed render function mapping (rays) -> pytree.
      rays: a `Rays` pytree, the rays to be rendered.
      config: A Config class.
      verbose: print progress indicators.

    Returns:
      rgb: torch.ndarray, rendered color image.
      disp: torch.ndarray, rendered disparity image.
      acc: torch.ndarray, rendered accumulated weights per pixel.
    """
    height, width = rays.origins.shape[:2]
    num_rays = height * width
    chunks = []
    idx0s = range(0, num_rays, config.render_chunk_size)
    for i_chunk, idx0 in enumerate(idx0s):
        # pylint: disable=cell-var-from-loop
        if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
            print(f'Rendering chunk {i_chunk}/{len(idx0s)-1}')
        chunk_rays = rays[idx0:idx0 + config.render_chunk_size]
        chunk_renderings, _ = render_fn(chunk_rays)

        # Gather the final pass for 2D buffers and all passes for ray bundles.
        chunk_rendering = chunk_renderings[-1]
        for k in chunk_renderings[0]:
            if k.startswith('ray_'):
                chunk_rendering[k] = [r[k] for r in chunk_renderings]

        chunks.append(chunk_rendering)

    # Concatenate all chunks within each leaf of a single pytree.
    rendering = torch.concatenate(chunks)
    for k, z in rendering.items():
        if not k.startswith('ray_'):
            # Reshape 2D buffers into original image shape.
            rendering[k] = z.reshape((height, width) + z.shape[1:])

    return rendering
