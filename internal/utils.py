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

"""Utility functions."""

import enum
import os
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image
import torch
from dataclasses import dataclass


_Array = Union[np.ndarray, torch.tensor]


@dataclass
class Pixels:
  """All tensors must have the same num_dims and first n-1 dims must match."""
  pix_x_int: _Array
  pix_y_int: _Array
  lossmult: _Array
  near: _Array
  far: _Array
  cam_idx: _Array

@dataclass
class Rays:
  """All tensors must have the same num_dims and first n-1 dims must match."""
  origins: _Array
  directions: _Array
  viewdirs: _Array
  radii: _Array
  imageplane: _Array
  lossmult: _Array
  near: _Array
  far: _Array
  cam_idx: _Array


# Dummy Rays object that can be used to initialize NeRF model.
def dummy_rays() -> Rays:
  data_fn = lambda n: torch.zeros((1, n))
  return Rays(
      origins=data_fn(3),
      directions=data_fn(3),
      viewdirs=data_fn(3),
      radii=data_fn(1),
      imageplane=data_fn(2),
      lossmult=data_fn(1),
      near=data_fn(1),
      far=data_fn(1),
      cam_idx=data_fn(1).type(torch.int32))


@dataclass
class Batch:
  """Data batch for NeRF training or testing."""
  rays: Union[Pixels, Rays]
  rgb: Optional[_Array] = None
  disps: Optional[_Array] = None
  normals: Optional[_Array] = None
  alphas: Optional[_Array] = None


class DataSplit(enum.Enum):
  """Dataset split."""
  TRAIN = 'train'
  TEST = 'test'


class BatchingMethod(enum.Enum):
  """Draw rays randomly from a single image or all images, in each batch."""
  ALL_IMAGES = 'all_images'
  SINGLE_IMAGE = 'single_image'


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return os.path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return os.path.isdir(pth)


def makedirs(pth):
  if not file_exists(pth):
    os.makedirs(pth)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y


def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def save_img_u8(img, pth, mask=None):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    img_np = (np.clip(np.nan_to_num(img), 0., 1.) * 255).astype(np.uint8).squeeze()
    if mask is not None:
      mask_np = (np.nan_to_num(mask)).astype(np.float32).squeeze()
      mask_np = 255 * (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())
      img_np = (255 - mask_np) + img_np
      img_np = np.array((255 * (img_np - img_np.min()) / (img_np.max() - img_np.min())), dtype=np.uint8)

    Image.fromarray(img_np).save(f, 'PNG')

def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
