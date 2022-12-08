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

"""Training script."""

import functools
import os
import sys
import gc
import time
import numpy as np
import random
import torch
import flatdict
import logging.config
from absl import flags
import absl

from torch.utils.tensorboard import SummaryWriter
from absl import app
import gin.torch
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis

FLAGS = flags.FLAGS
configs.define_common_flags()
TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


def main(unused_argv):
    config = configs.load_config()

    # setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    # set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # load training and test sets
    dataset = datasets.load_dataset('train', config.data_dir, config)
    test_dataset = datasets.load_dataset('test', config.data_dir, config)

    # create model, state, rendering evaluation function, training step, and lr scheduler
    setup = train_utils.setup_model(config, dataset=dataset)
    model, optimizer, lr_scheduler, render_eval_fn, train_step = setup
    state = dict(
        step=0,
        model=model.state_dict(),
        optim=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict(),
    )

    # create object for calculating metrics
    metric_harness = image.MetricHarness()

    # load saved checkpoint or create checkpoint dir if not there
    if utils.isdir(config.checkpoint_dir):
        files = sorted([f for f in os.listdir(config.checkpoint_dir)
                 if f.startswith('checkpoint')], key=lambda x: x.split('_')[-1])
        # if there are checkpoints in the dir, load the latest checkpoint
        if files:
            checkpoint_name = files[-1]
            state = torch.load(os.path.join(config.checkpoint_dir, checkpoint_name))
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            lr_scheduler.load_state_dict(state['lr_scheduler'])
    else:
        utils.makedirs(config.checkpoint_dir)

    # setup logging to file
    logfile = os.path.join(config.checkpoint_dir, 'output.log')
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)])

    # print the number of parameters of the model
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Number of parameters being optimized: {num_params}')

    # Resume training at the step of the last checkpoint.
    init_step = state['step'] + 1

    # setup tensorboard for logging
    summary_writer = SummaryWriter(config.checkpoint_dir)

    # Prefetch_buffer_size = 3 x batch_size.
    # gc.disable()  # Disable automatic garbage collection for efficiency.
    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_steps
    else:
        num_steps = config.max_steps

    # set model to training mode and send to device
    model.to(device)

    # start training loop
    for step, batch in zip(range(init_step, num_steps + 1), dataset):

        model.train()

        # clear stats for this iteration
        if reset_stats:
            stats_buffer = []
            train_start_time = time.time()
            reset_stats = False

        # update fraction of completed training
        train_frac = np.clip((step - 1) / (config.max_steps - 1), 0, 1)

        # perform training step
        stats = train_step(
            model,
            optimizer,
            lr_scheduler,
            batch,
            dataset.cameras,
            train_frac,
        )

        # if step % config.gc_every == 0:
            # Disable automatic garbage collection for efficiency.
            # gc.collect()

        #TODO: Redundant?
        del batch
        gc.collect()
        torch.cuda.empty_cache()

        # set model to inference mode
        model.eval()

        with torch.no_grad():

            # Log training summaries
            stats_buffer.append(stats)

            if step == init_step or step % config.print_every == 0:

                elapsed_time = time.time() - train_start_time
                steps_per_sec = config.print_every / elapsed_time
                rays_per_sec = config.batch_size * steps_per_sec

                # A robust approximation of total training time, in case of pre-emption.
                total_time += int(round(TIME_PRECISION * elapsed_time))
                total_steps += config.print_every
                approx_total_time = int(round(step * total_time / total_steps))

                # Stack stats_buffer along axis 0.
                fs = [dict(flatdict.FlatDict(s, delimiter='/')) for s in stats_buffer]
                stats_stacked = {k: torch.stack([f[k] for f in fs]) for k in fs[0].keys()}

                # Split every statistic that isn't a vector into a set of statistics.
                stats_split = {}
                for k, v in stats_stacked.items():
                    if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
                        raise ValueError('statistics must be of size [n], or [n, k].')
                    if v.ndim == 1:
                        stats_split[k] = v
                    elif v.ndim == 2:
                        for i, vi in enumerate(tuple(v.T)):
                            stats_split[f'{k}/{i}'] = vi

                # Summarize the entire histogram of each statistic.
                for k, v in stats_split.items():
                    summary_writer.add_histogram('train_' + k, v, step)

                # Take the mean and max of each statistic since the last summary.
                avg_stats = {k: torch.mean(v) for k, v in stats_split.items()}
                max_stats = {k: torch.max(v) for k, v in stats_split.items()}

                # Summarize the mean and max of each statistic.
                for k, v in avg_stats.items():
                    summary_writer.add_scalar(f'train_avg_{k}', v, step)
                for k, v in max_stats.items():
                    summary_writer.add_scalar(f'train_max_{k}', v, step)
                summary_writer.add_scalar('train_num_params', num_params, step)
                summary_writer.add_scalar('train_learning_rate', *lr_scheduler.get_last_lr(), step)

                summary_writer.add_scalar('train_steps_per_sec', steps_per_sec, step)
                summary_writer.add_scalar('train_rays_per_sec', rays_per_sec, step)
                summary_writer.add_scalar('train_avg_psnr_timed', avg_stats['psnr'],
                                    total_time // TIME_PRECISION)
                summary_writer.add_scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                                    approx_total_time // TIME_PRECISION)
                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                avg_loss = avg_stats['loss']
                avg_psnr = avg_stats['psnr']
                str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
                    k[7:11]: (f'{v:0.5f}' if v >= 1e-4 and v < 10 else f'{v:0.1e}')
                    for k, v in avg_stats.items()
                    if k.startswith('losses/')
                }
                logging.info(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
                    f'loss={avg_loss:0.5f}, ' + f'psnr={avg_psnr:6.3f}, ' +
                    f'lr={lr_scheduler.get_last_lr()[0]:0.2e} | ' +
                    ', '.join([f'{k}={s}' for k, s in str_losses.items()]) +
                    f', {rays_per_sec:0.0f} r/s')

                # Reset everything we are tracking between summarizations.
                reset_stats = True

            # save a checkpoint on the first epoch and every Nth epoch
            if step == 1 or step % config.checkpoint_every == 0:
                # save checkpoint
                state = dict(
                    step=step,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict())
                torch.save(state, os.path.join(config.checkpoint_dir, f'checkpoint_{step}'))

            # Test-set evaluation.
            if config.train_render_every > 0 and step % config.train_render_every == 0:
                # We reuse the same random number generator from the optimization step
                # here on purpose so that the visualization matches what happened in
                # training.
                eval_start_time = time.time()
                test_case = next(test_dataset)
                test_case.rays.to(device)

                # render test image
                rendering = models.render_image(
                    functools.partial(render_eval_fn, train_frac),
                    test_case.rays, config)

                # Log eval summaries
                eval_time = time.time() - eval_start_time
                num_rays = np.prod(np.array(test_case.rays.directions.shape[:-1]))
                rays_per_sec = num_rays / eval_time
                summary_writer.add_scalar('test_rays_per_sec', rays_per_sec, step)
                logging.info(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')

                if config.compute_eval_metrics:
                    metric_start_time = time.time()
                    metric = metric_harness(rendering['rgb'], test_case.rgb)
                    logging.info(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
                    for name, val in metric.items():
                        if not np.isnan(val):
                            logging.info(f'{name} = {val:.4f}')
                            summary_writer.add_scalar(
                                'train_metrics/' + name, val, step)

                vis_start_time = time.time()
                vis_suite = vis.visualize_suite(rendering, test_case.rays)
                logging.info(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
                summary_writer.add_image(
                    'test_true_color', test_case.rgb, step, dataformats='HWC')
                if config.compute_normal_metrics:
                    summary_writer.add_image(
                        'test_true_normals', test_case.normals / 2. + 0.5, step,
                        dataformats='HWC')
                for k, v in vis_suite.items():
                    summary_writer.add_image(
                        'test_output_' + k, v, step,
                        dataformats='HWC' if len(v.shape) == 3 else 'HW')

        # save last checkpoint if it wasn't already saved
        if config.max_steps % config.checkpoint_every != 0:
            state = dict(
                step=config.max_steps,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict())
            torch.save(state, os.path.join(
                config.checkpoint_dir, f'checkpoint_{config.max_steps}'))


if __name__ == '__main__':
    with gin.config_scope('train'):
        # app.run(main)
        FLAGS(sys.argv)
        main(sys.argv)
