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

"""Training step and model creation functions."""

import collections
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Text, Tuple
import torch

from internal import camera_utils
from internal import configs
from internal import datasets
from internal import image
from internal import math
from internal import models
from internal import ref_utils
from internal import stepfun
from internal import utils


def compute_data_loss(batch, renderings, rays, config):
    """Computes data loss terms for RGB, normal, and depth outputs."""
    data_losses = []
    stats = collections.defaultdict(lambda: [])

    # lossmult can be used to apply a weight to each ray in the batch.
    # For example: masking out rays, applying the Bayer mosaic mask, upweighting
    # rays from lower resolution images and so on.
    lossmult = rays.lossmult
    lossmult = torch.broadcast_to(lossmult, batch.rgb[..., :3].shape)
    if config.disable_multiscale_loss:
        lossmult = torch.ones_like(lossmult)
    for rendering in renderings:
        resid_sq = (rendering['rgb'] - torch.tensor(batch.rgb[..., :3]))**2
        denom = lossmult.sum()
        stats['mses'].append((lossmult * resid_sq).sum() / denom)

        if config.data_loss_type == 'mse':
            # Mean-squared error (L2) loss.
            data_loss = resid_sq
        elif config.data_loss_type == 'charb':
            # Charbonnier loss.
            data_loss = torch.sqrt(resid_sq + config.charb_padding**2)
        else:
            assert False
        data_losses.append((lossmult * data_loss).sum() / denom)

        if config.compute_disp_metrics:
            # Using mean to compute disparity, but other distance statistics can
            # be used instead.
            disp = 1 / (1 + rendering['distance_mean'])
            stats['disparity_mses'].append(((disp - batch.disps)**2).mean())

        if config.compute_normal_metrics:
            if 'normals' in rendering:
                weights = rendering['acc'] * batch.alphas
                normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
                normalized_normals = ref_utils.l2_normalize(
                    rendering['normals'])
                normal_mae = ref_utils.compute_weighted_mae(weights, normalized_normals,
                                                            normalized_normals_gt)
            else:
                # If normals are not computed, set MAE to NaN.
                normal_mae = torch.nan

            stats['normal_maes'].append(normal_mae)

    data_losses = torch.stack(data_losses)
    loss = \
        config.data_coarse_loss_mult * torch.sum(data_losses[:-1]) + \
        config.data_loss_mult * data_losses[-1]
    stats = {k: torch.tensor(stats[k]) for k in stats}
    return loss, stats


def orientation_loss(rays, model, ray_history, config):
    """Computes the orientation loss regularizer defined in ref-NeRF."""
    total_loss = 0.
    zero = torch.tensor(0.0, dtype=torch.float32)
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results[config.orientation_loss_target]
        if n is None:
            raise ValueError(
                'Normals cannot be None if orientation loss is on.')
        # Negate viewdirs to represent normalized vectors from point to camera.
        v = -1. * rays.viewdirs
        n_dot_v = (n * v[..., None, :]).sum(axis=-1)
        loss = torch.mean((w * torch.minimum(zero, n_dot_v)**2).sum(axis=-1))
        if i < model.num_levels - 1:
            total_loss += config.orientation_coarse_loss_mult * loss
        else:
            total_loss += config.orientation_loss_mult * loss
    return total_loss


def predicted_normal_loss(model, ray_history, config):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    total_loss = 0.
    for i, ray_results in enumerate(ray_history):
        w = ray_results['weights']
        n = ray_results['normals']
        n_pred = ray_results['normals_pred']
        if n is None or n_pred is None:
            raise ValueError(
                'Predicted normals and gradient normals cannot be None if '
                'predicted normal loss is on.')
        loss = torch.mean(
            (w * (1.0 - torch.sum(n * n_pred, axis=-1))).sum(axis=-1))
        if i < model.num_levels - 1:
            total_loss += config.predicted_normal_coarse_loss_mult * loss
        else:
            total_loss += config.predicted_normal_loss_mult * loss
    return total_loss


def create_train_step(model: models.Model,
                      config: configs.Config,
                      dataset: Optional[datasets.Dataset] = None):
    """Creates the pmap'ed Nerf training function.

    Args:
      model: The linen model.
      config: The configuration.
      dataset: Training dataset.

    Returns:
      training function.
    """
    if dataset is None:
        camtype = camera_utils.ProjectionType.PERSPECTIVE
    else:
        camtype = dataset.camtype

    def train_step(
        model,
        optimizer,
        lr_scheduler,
        batch,
        cameras,
        train_frac,
    ):
        """One optimization step.

        Args:
          state: TrainState, state of the model/optimizer.
          batch: dict, a mini-batch of data for training.
          cameras: module containing camera poses.
          train_frac: float, the fraction of training that is complete.

        Returns:
          A tuple (new_state, stats) with
            new_state: TrainState, new training state.
            stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
        """
        rays = batch.rays
        if config.cast_rays_in_train_step:
            rays = camera_utils.cast_ray_batch(
                cameras, rays, camtype, xnp=torch).to(device)
        else:
            rays.to(model.device)

        # Indicates whether we need to compute output normal or depth maps in 2D.
        compute_extras = (
            config.compute_disp_metrics or config.compute_normal_metrics)

        # clear gradients
        optimizer.zero_grad()

        renderings, ray_history = model(
            rays,
            train_frac=train_frac,
            compute_extras=compute_extras)

        losses = {}

        # calculate photometric error
        data_loss, stats = compute_data_loss(batch, renderings, rays, config)
        losses['data'] = data_loss

        # calculate normals orientation loss
        if (config.orientation_coarse_loss_mult > 0 or
                config.orientation_loss_mult > 0):
            losses['orientation'] = orientation_loss(
                rays, model, ray_history, config)

        # calculate predicted normal loss
        if (config.predicted_normal_coarse_loss_mult > 0 or
                config.predicted_normal_loss_mult > 0):
            losses['predicted_normals'] = predicted_normal_loss(
                model, ray_history, config)

        params = dict(model.named_parameters())
        stats['weights_l2s'] = {k.replace('.', '/') : params[k].detach().norm() ** 2 for k in params}

        # calculate total loss
        loss = torch.sum(torch.stack(list(losses.values())))
        stats['loss'] = loss
        stats['losses'] = losses

        # backprop
        loss.backward()

        # import pdb
        # pdb.set_trace()
        # pdb.pm()

        # calculate average grad and stats
        stats['grad_norms'] = {k.replace('.', '/') : params[k].grad.detach().norm() for k in params}
        stats['grad_maxes'] = {k.replace('.', '/') : params[k].grad.detach().abs().max() for k in params}

        # Clip gradients
        if config.grad_max_val > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config.grad_max_value)
        if config.grad_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_max_norm)
        #TODO: set nan grads to 0
        # grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)

        # update the model weights
        optimizer.step()

        # update learning rate
        lr_scheduler.step()

        #TODO: difference between previous and current state - Redundant?
        # stats['opt_update_norms'] = summarize_tree(opt_delta, tree_norm)
        # stats['opt_update_maxes'] = summarize_tree(opt_delta, tree_abs_max)

        # Calculate PSNR metric
        stats['psnrs'] = image.mse_to_psnr(stats['mses'])
        stats['psnr'] = stats['psnrs'][-1]

        # return new state and statistics
        return stats

    return train_step


def create_optimizer(
        config: configs.Config,
        params: Dict) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Creates optimizer for model training."""
    adam_kwargs = {
        'lr': config.lr_init,
        'betas': (config.adam_beta1, config.adam_beta2),
        'eps': config.adam_eps,
    }
    lr_kwargs = {
        'max_steps': config.max_steps,
        'lr_delay_steps': config.lr_delay_steps,
        'lr_delay_mult': config.lr_delay_mult,
    }

    def get_lr_fn(lr_init, lr_final):
        return functools.partial(
            math.learning_rate_decay,
            lr_init=lr_init,
            lr_final=lr_final,
            **lr_kwargs)

    optimizer = torch.optim.Adam(params=params, **adam_kwargs)
    lr_fn_main = get_lr_fn(config.lr_init, config.lr_final)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn_main)

    return optimizer, lr_scheduler


def create_render_fn(model: models.Model):
    """Creates a function for full image rendering."""
    def render_eval_fn(train_frac, rays):
        return model(
            rays,
            train_frac=train_frac,
            compute_extras=True)
    return render_eval_fn


def setup_model(
        config: configs.Config,
        dataset: Optional[datasets.Dataset] = None,
    ):
    """Creates NeRF model, optimizer, and pmap-ed train/render functions."""

    dummy_rays = utils.dummy_rays()
    model = models.construct_model(dummy_rays, config)

    optimizer, lr_scheduler = create_optimizer(config, model.parameters())
    render_eval_fn = create_render_fn(model)
    train_step = create_train_step(model, config, dataset=dataset)

    return model, optimizer, lr_scheduler, render_eval_fn, train_step
