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
import gc
import time
import torch
from absl import app
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
import numpy as np

configs.define_common_flags()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


def main(unused_argv):
    config = configs.load_config()

    # load training and test sets
    dataset = datasets.load_dataset('train', config.data_dir, config)
    test_dataset = datasets.load_dataset('test', config.data_dir, config)

    # create model, state, rendering evaluation function, training step, and lr scheduler
    setup = train_utils.setup_model(config, dataset=dataset)
    model, variables, optimizer, lr_scheduler, render_eval_fn, train_step = setup
    state = {
        'step': 1,
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    num_params = np.prod(variables.shape)
    print(f'Number of parameters being optimized: {num_params}')

    # create object for calculating metrics
    metric_harness = image.MetricHarness()

    # load saved checkpoint or create checkpoint dir if not there
    if utils.isdir(config.checkpoint_dir):
        state = torch.load(config.checkpoint_dir)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        lr_scheduler.load_state_dict(state['lr_scheduler'])
    else:
        utils.makedirs(config.checkpoint_dir)

    # Resume training at the step of the last checkpoint.
    init_step = state.step + 1

    # setup tensorboard for logging
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)

    # Prefetch_buffer_size = 3 x batch_size.
    # gc.disable()  # Disable automatic garbage collection for efficiency.
    total_time = 0
    total_steps = 0
    reset_stats = True
    if config.early_exit_steps is not None:
        num_steps = config.early_exit_steps
    else:
        num_steps = config.max_steps

    # set model to training mode
    model.train()

    # start training loop
    for step, batch in zip(range(init_step, num_steps + 1), dataset):

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

        if step % config.gc_every == 0:
            # Disable automatic garbage collection for efficiency.
            gc.collect()

        # Log training summaries
        stats_buffer.append(stats)

        with torch.no_grad():

            if step == init_step or step % config.print_every == 0:

                elapsed_time = time.time() - train_start_time
                steps_per_sec = config.print_every / elapsed_time
                rays_per_sec = config.batch_size * steps_per_sec

                # A robust approximation of total training time, in case of pre-emption.
                total_time += int(round(TIME_PRECISION * elapsed_time))
                total_steps += config.print_every
                approx_total_time = int(round(step * total_time / total_steps))

                # Transpose and stack stats_buffer along axis 0.
                # fs = [flax.traverse_util.flatten_dict(s, sep='/') for s in stats_buffer]  #TODO:
                fs = stats_buffer
                stats_stacked = {k: np.stack([f[k] for f in fs]) for k in fs[0].keys()}

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
                    summary_writer.histogram('train_' + k, v, step)

                # Take the mean and max of each statistic since the last summary.
                avg_stats = {k: jnp.mean(v) for k, v in stats_split.items()}
                max_stats = {k: jnp.max(v) for k, v in stats_split.items()}

                # Summarize the mean and max of each statistic.
                for k, v in avg_stats.items():
                    summary_writer.scalar(f'train_avg_{k}', v, step)
                for k, v in max_stats.items():
                    summary_writer.scalar(f'train_max_{k}', v, step)

                summary_writer.scalar('train_num_params', num_params, step)
                summary_writer.scalar('train_learning_rate', learning_rate, step)
                summary_writer.scalar('train_steps_per_sec', steps_per_sec, step)
                summary_writer.scalar('train_rays_per_sec', rays_per_sec, step)
                summary_writer.scalar('train_avg_psnr_timed', avg_stats['psnr'],
                                    total_time // TIME_PRECISION)
                summary_writer.scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                                    approx_total_time // TIME_PRECISION)
                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                avg_loss = avg_stats['loss']
                avg_psnr = avg_stats['psnr']
                str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
                    k[7:11]: (f'{v:0.5f}' if v >= 1e-4 and v < 10 else f'{v:0.1e}')
                    for k, v in avg_stats.items()
                    if k.startswith('losses/')
                }
                print(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
                    f'loss={avg_loss:0.5f}, ' + f'psnr={avg_psnr:6.3f}, ' +
                    f'lr={learning_rate:0.2e} | ' +
                    ', '.join([f'{k}={s}' for k, s in str_losses.items()]) +
                    f', {rays_per_sec:0.0f} r/s')

                # Reset everything we are tracking between summarizations.
                reset_stats = True

            # save a checkpoint on the first epoch and every Nth epoch
            if step == 1 or step % config.checkpoint_every == 0:
                # save checkpoint
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, path=os.path.join(config.checkpoint_dir, f'checkpoint_{step}'))

            # Test-set evaluation.
            if config.train_render_every > 0 and step % config.train_render_every == 0:
                # We reuse the same random number generator from the optimization step
                # here on purpose so that the visualization matches what happened in
                # training.
                eval_start_time = time.time()
                eval_variables = state.params
                test_case = next(test_dataset)
                rendering = models.render_image(
                    functools.partial(render_eval_fn, eval_variables, train_frac),
                    test_case.rays, config)

                # Log eval summaries
                eval_time = time.time() - eval_start_time
                num_rays = np.prod(np.array(test_case.rays.directions.shape[:-1]))
                rays_per_sec = num_rays / eval_time
                summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
                print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')

                if config.compute_eval_metrics:
                    metric_start_time = time.time()
                    metric = metric_harness(rendering['rgb'], test_case.rgb)
                    print(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
                    for name, val in metric.items():
                        if not np.isnan(val):
                            print(f'{name} = {val:.4f}')
                            summary_writer.scalar(
                                'train_metrics/' + name, val, step)

                vis_start_time = time.time()
                vis_suite = vis.visualize_suite(rendering, test_case.rays)
                print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
                summary_writer.image('test_true_color', test_case.rgb, step)
                if config.compute_normal_metrics:
                    summary_writer.image('test_true_normals',
                                        test_case.normals / 2. + 0.5, step)
                for k, v in vis_suite.items():
                    summary_writer.image('test_output_' + k, v, step)

        # save last checkpoint if it wasn't already saved
        if config.max_steps % config.checkpoint_every != 0:
            # TODO: use pytorch based checkpointing
                torch.save({
                    'step': config.max_steps,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, path=os.path.join(config.checkpoint_dir, f'checkpoint_{config.max_steps}'))


if __name__ == '__main__':
    with gin.config_scope('train'):
        app.run(main)
