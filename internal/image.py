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

"""Functions for processing images."""

import types
from typing import Optional, Union
import torch
import torch.nn.functional as F

# import dm_pix
import numpy as np
from numpy import array as tensor

_Array = Union[np.ndarray, torch.tensor]
np.tensor = np.array


def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / torch.log(torch.tensor(10.)) * torch.log(mse)


def psnr_to_mse(psnr):
    """Compute MSE given a PSNR (we assume the maximum pixel value is 1)."""
    return torch.exp(-0.1 * torch.log(torch.tensor(10.)) * psnr)


def ssim_to_dssim(ssim):
    """Compute DSSIM given an SSIM."""
    return (1 - ssim) / 2


def dssim_to_ssim(dssim):
    """Compute DSSIM given an SSIM."""
    return 1 - 2 * dssim


def linear_to_srgb(linear: _Array,
                   eps: Optional[float] = None,
                   xnp: types.ModuleType = torch) -> _Array:
    """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = xnp.tensor(xnp.finfo(xnp.float32).eps)
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * xnp.maximum(eps, linear) ** (5 / 12) - 11) / 200
    return xnp.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb: _Array,
                   eps: Optional[float] = None,
                   xnp: types.ModuleType = torch) -> _Array:
    """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
    if eps is None:
        eps = xnp.tensor(xnp.finfo(xnp.float32).eps)
    linear0 = 25 / 323 * srgb
    linear1 = xnp.maximum(eps, ((200 * srgb + 11) / (211))) ** (12 / 5)
    return xnp.where(srgb <= 0.04045, linear0, linear1)


def downsample(img, factor):
    """Area downsample img (factor must evenly divide img height and width)."""
    sh = img.shape
    if not (sh[0] % factor == 0 and sh[1] % factor == 0):
        raise ValueError(f'Downsampling factor {factor} does not '
                         f'evenly divide image shape {sh[:2]}')
    img = img.reshape((sh[0] // factor, factor, sh[1] // factor, factor) + sh[2:])
    img = img.mean((1, 3))
    return img


def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, 0)
            mb = torch.where(mask, b, 0)
            # Solve the linear system. We're using the np.lstsq instead of torch because
            # it's significantly more stable in this case, for some reason.
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(
            torch.matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img


class MetricHarness:
    """A helper class for evaluating several error metrics."""

    def __init__(self):
        pass

    def __call__(self, rgb_pred, rgb_gt, name_fn=lambda s: s):
        """Evaluate the error between a predicted rgb image and the true image."""
        psnr = float(mse_to_psnr(((rgb_pred - rgb_gt) ** 2).mean()))
        ssim = float(self.compute_ssim(rgb_pred, rgb_gt))

        return {
            name_fn('psnr'): psnr,
            name_fn('ssim'): ssim,
        }

    @staticmethod
    def compute_ssim(img0, img1, max_val=1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, return_map=False):
        """Computes SSIM from two images.
        This function was modeled after tf.image.ssim, and should produce comparable
        output.
        Args:
          img0: torch.tensor. An image of size [..., width, height, num_channels].
          img1: torch.tensor. An image of size [..., width, height, num_channels].
          max_val: float > 0. The maximum magnitude that `img0` or `img1` can have.
          filter_size: int >= 1. Window size.
          filter_sigma: float > 0. The bandwidth of the Gaussian used for filtering.
          k1: float > 0. One of the SSIM dampening parameters.
          k2: float > 0. One of the SSIM dampening parameters.
          return_map: Bool. If True, will cause the per-pixel SSIM "map" to returned
        Returns:
          Each image's mean SSIM, or a tensor of individual values if `return_map`.
        """
        device = img0.device
        ori_shape = img0.size()
        width, height, num_channels = ori_shape[-3:]
        img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
        img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
        batch_size = img0.shape[0]
        dtype = img0.dtype

        # Construct a 1D Gaussian blur filter.
        hw = filter_size // 2
        shift = (2 * hw - filter_size + 1) / 2
        f_i = ((torch.arange(filter_size, device=device, dtype=dtype) - hw + shift) / filter_sigma) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)

        # Blur in x and y (faster than the 2D convolution).
        # z is a tensor of size [B, H, W, C]
        filt_fn1 = lambda z: F.conv2d(
            z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
            padding=[hw, 0], groups=num_channels)
        filt_fn2 = lambda z: F.conv2d(
            z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
            padding=[0, hw], groups=num_channels)

        # Vmap the blurs to the tensor size, and then compose them.
        filt_fn = lambda z: filt_fn1(filt_fn2(z))
        mu0 = filt_fn(img0)
        mu1 = filt_fn(img1)
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(img0 ** 2) - mu00
        sigma11 = filt_fn(img1 ** 2) - mu11
        sigma01 = filt_fn(img0 * img1) - mu01

        # Clip the variances and covariances to valid values.
        # Variance must be non-negative:
        sigma00 = torch.clamp(sigma00, min=0.0)
        sigma11 = torch.clamp(sigma11, min=0.0)
        sigma01 = torch.sign(sigma01) * torch.min(
            torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
        )

        c1 = (k1 * max_val) ** 2
        c2 = (k2 * max_val) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom
        ssim = torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1)
        return ssim_map if return_map else ssim
