# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DP-FTRL training, based on paper
"Practical and Private (Deep) Learning without Sampling or Shuffling"
https://arxiv.org/abs/2103.00039.
"""

from absl import app
from absl import flags

import os
from tqdm import trange
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from privacy import compute_epsilon_tree, compute_epsilon_dpsgd_noamp

from optimizers import FTRLOptimizer
from ftrl_noise import CummuNoiseTorch, CummuNoiseEffTorch
from nn import get_nn
from datasets import get_data
from random_batch import make_train_loader
import utils
from utils import EasyDict


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'mnist', ['mnist', 'cifar10', 'emnist_merge'], '')

flags.DEFINE_enum(
    'method',
    'dp_ftrl',
    ['dp_ftrl', 'dp_sgd_amp', 'dp_sgd_noamp'],
    'Training method. dp_ftrl: DP-FTRL/DP-FTRLM (tree noise). '
    'dp_sgd_amp: DP-SGD with privacy amplification via subsampling (Opacus accountant). '
    'dp_sgd_noamp: DP-SGD baseline without amplification (composed Gaussian).'
)

# Backward-compat: keep the old flag name. If explicitly set, it overrides method.
flags.DEFINE_boolean('dp_ftrl', True, 'DEPRECATED. Use --method. If True -> dp_ftrl, else -> dp_sgd_noamp.')
flags.DEFINE_float('noise_multiplier', 4.0, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_integer('restart', 0, 'If > 0, restart the tree every this number of epoch(s).')
flags.DEFINE_boolean('effi_noise', False, 'If True, use tree aggregation proposed in https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf.')
flags.DEFINE_boolean('tree_completion', False, 'If true, generate until reaching a power of 2.')

flags.DEFINE_float('momentum', 0, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 0.4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 250, 'Batch size.')
flags.DEFINE_integer('epochs', 3, 'Number of epochs.')

flags.DEFINE_boolean('random_batch', False, 'If True, use variable batch sizes via truncated geometric noise.')
flags.DEFINE_float('random_batch_delta', 1e-5, 'Delta parameter used to truncate the geometric noise (only if random_batch=True).')

flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')

flags.DEFINE_integer('run', 1, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')


def main(argv):
    # PyTorch-only: no TensorFlow / TFDS usage.

    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Resolve training method (backward compatibility with --dp_ftrl).
    method = FLAGS.method
    try:
        if FLAGS['method'].present is False and FLAGS['dp_ftrl'].present is True:
            method = 'dp_ftrl' if FLAGS.dp_ftrl else 'dp_sgd_noamp'
    except Exception:
        # If the flag library changes, fall back to --method.
        method = FLAGS.method

    if method == 'dp_sgd_amp' and FLAGS.random_batch:
        print('[WARN] --method=dp_sgd_amp assumes uniform random subsampling (privacy amplification). '
              'With --random_batch enabled the sampling is no longer the paper baseline. '
              'Switching to dp_sgd_noamp for a conservative guarantee.')
        method = 'dp_sgd_noamp'

    is_ftrl = (method == 'dp_ftrl')
    is_sgd = (method in ('dp_sgd_amp', 'dp_sgd_noamp'))

    # Data
    train_ds, test_ds, ntrain, nclass = get_data(FLAGS.data)
    print('Training set size', ntrain)

    # Hyperparameters for training.
    epochs = FLAGS.epochs
    batch = FLAGS.batch_size if FLAGS.batch_size > 0 else ntrain
    # With random_batch, number of batches per epoch is variable; we'll iterate the loader.
    num_batches = ntrain // batch
    noise_multiplier = FLAGS.noise_multiplier
    clip = FLAGS.l2_norm_clip
    lr = FLAGS.learning_rate
    if not FLAGS.restart or not is_ftrl:
        FLAGS.tree_completion = False

    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    assert report_nimg % batch == 0

    # Get the name of the output directory.
    log_dir = os.path.join(
        FLAGS.dir,
        FLAGS.data,
        utils.get_fn(
            EasyDict(batch=batch),
            EasyDict(method=method, restart=FLAGS.restart, completion=FLAGS.tree_completion,
                     noise=noise_multiplier, clip=clip, mb=1, random_batch=FLAGS.random_batch),
            [
                EasyDict({'lr': lr}),
                EasyDict(m=FLAGS.momentum if FLAGS.momentum > 0 else None,
                         effi=FLAGS.effi_noise if is_ftrl else None),
                EasyDict(sd=FLAGS.run)
            ],
        ),
    )
    print('Model dir', log_dir)

    # DataLoader (optionally with random/variable batch sizes).
    train_loader = make_train_loader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        random_batch=FLAGS.random_batch,
        delta=FLAGS.random_batch_delta if FLAGS.random_batch else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Function to conduct training for one epoch
    def train_loop(model, device, optimizer, cumm_noise, epoch, writer):
        model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        losses = []
        # We track how many examples have been processed for reporting.
        processed = 0
        step = epoch * (len(train_loader) if hasattr(train_loader, '__len__') else 0)
        for data, target in train_loader:
            step += 1
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            per_example_loss = criterion(output, target)
            # IMPORTANT:
            # - Opacus' DPOptimizer (used only for clipping here) scales gradients by
            #   1/expected_batch_size internally.
            # - To get gradients proportional to sum(loss_i) / mean_batch_size, we
            #   backprop through sum(loss_i) and let the optimizer handle the division.
            # - If Opacus is NOT attached, we manually scale by mean batch size.
            if is_ftrl:
                loss = per_example_loss.sum()
            else:
                # DP-SGD expects mean loss for fixed batch. For random batches we
                # backprop through sum(loss_i) and let DPOptimizer divide by the
                # *expected* batch size (mean batch size).
                loss = per_example_loss.sum() if FLAGS.random_batch else per_example_loss.mean()
            loss.backward()

            if is_ftrl:
                optimizer.step((lr, cumm_noise()))
            else:
                optimizer.step()
            losses.append(loss.item())

            processed += int(data.shape[0])

            if processed >= report_nimg:
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, step)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, step)
                model.train()
                print('Step %04d Accuracy %.2f' % (step, 100 * acc_test))
                processed = 0

        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print('Epoch %04d Loss %.2f' % (epoch + 1, np.mean(losses)))

    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        with torch.no_grad():
            accs = [0, 0]
            # Train accuracy (single pass)
            n_seen = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                accs[0] += pred.eq(target.view_as(pred)).sum().item()
                n_seen += int(data.shape[0])
            accs[0] /= max(1, n_seen)

            # Test accuracy
            n_seen = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                accs[1] += pred.eq(target.view_as(pred)).sum().item()
                n_seen += int(data.shape[0])
            accs[1] /= max(1, n_seen)
        return accs

    # Get model for different dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_nn(
        {'mnist': 'small_nn', 'emnist_merge': 'small_nn', 'cifar10': 'vgg128'}[FLAGS.data],
        nclass=nclass,
    )

    # Opacus requires modules to be compatible with per-sample gradients.
    if (is_ftrl or is_sgd) and (not ModuleValidator.is_valid(model)):
        model = ModuleValidator.fix(model)
    model = model.to(device)

    # Delta choices as used in the paper's centralized experiments.
    delta = {'mnist': 1e-5, 'cifar10': 1e-5, 'emnist_merge': 1e-6}[FLAGS.data]

    privacy_engine = None
    if is_ftrl:
        # DP-FTRL/DP-FTRLM:
        # 1) Use Opacus ONLY for per-sample gradient clipping (noise_multiplier=0).
        # 2) Generate DP noise using tree aggregation (CummuNoise*) and pass it to the FTRL optimizer.
        optimizer = FTRLOptimizer(
            model.parameters(),
            momentum=FLAGS.momentum,
            record_last_noise=FLAGS.restart > 0 and FLAGS.tree_completion,
        )
        privacy_engine = PrivacyEngine(
            model,
            batch_size=batch,
            sample_size=ntrain,
            alphas=[],
            noise_multiplier=0.0,
            max_grad_norm=clip,
        )
        privacy_engine.attach(optimizer)
    else:
        # DP-SGD baselines.
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=float(FLAGS.momentum))
        if method == 'dp_sgd_amp':
            # Faithful baseline: amplification via random subsampling.
            # Opacus accountant assumes Poisson/uniform subsampling; this matches the paper baseline
            # when --random_batch is False (enforced above).
            alphas = list(np.arange(1.1, 32.0, 0.1)) + list(range(32, 128))
            privacy_engine = PrivacyEngine(
                model,
                batch_size=batch,
                sample_size=ntrain,
                alphas=alphas,
                noise_multiplier=noise_multiplier,
                max_grad_norm=clip,
            )
            privacy_engine.attach(optimizer)
        else:
            # No-amplification baseline: use the same clipping/noise injection, but account privacy
            # conservatively as a composed Gaussian mechanism (no subsampling amplification).
            privacy_engine = PrivacyEngine(
                model,
                batch_size=batch,
                sample_size=batch,  # sample_rate=1 inside Opacus; we do our own accounting.
                alphas=[],
                noise_multiplier=noise_multiplier,
                max_grad_norm=clip,
            )
            privacy_engine.attach(optimizer)
    shapes = [p.shape for p in model.parameters()]

    def get_cumm_noise(effi_noise):
        if (not is_ftrl) or noise_multiplier == 0:
            return lambda: [torch.Tensor([0]).to(device)] * len(shapes)  # just return scalar 0
        if not effi_noise:
            cumm_noise = CummuNoiseTorch(noise_multiplier * clip / batch, shapes, device)
        else:
            cumm_noise = CummuNoiseEffTorch(noise_multiplier * clip / batch, shapes, device)
        return cumm_noise

    cumm_noise = get_cumm_noise(FLAGS.effi_noise)

    # The training loop.
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    for epoch in range(epochs):
        train_loop(model, device, optimizer, cumm_noise, epoch, writer)

        if epoch + 1 == epochs:
            break
        restart_now = is_ftrl and epoch < epochs - 1 and FLAGS.restart > 0 and (epoch + 1) % FLAGS.restart == 0
        if restart_now:
            last_noise = None
            if FLAGS.tree_completion:
                actual_steps = num_batches * FLAGS.restart
                next_pow_2 = 2**(actual_steps - 1).bit_length()
                if next_pow_2 > actual_steps:
                    last_noise = cumm_noise.proceed_until(next_pow_2)
            optimizer.restart(last_noise)
            cumm_noise = get_cumm_noise(FLAGS.effi_noise)
            # DataLoader shuffles per-epoch when shuffle=True. (DP-FTRL does not
            # require shuffling for privacy; this is purely an optimization choice.)
    writer.close()

    # Report privacy.
    try:
        if method == 'dp_sgd_amp' and privacy_engine is not None:
            eps = float(privacy_engine.get_epsilon(delta))
            print(f'[Privacy] DP-SGD (amp): (epsilon, delta)=({eps:.4f}, {delta})')
        elif method == 'dp_sgd_noamp':
            steps = epochs * num_batches
            eps = compute_epsilon_dpsgd_noamp(steps=steps, noise_multiplier=noise_multiplier, delta=delta)
            print(f'[Privacy] DP-SGD (no-amp): steps={steps}, (epsilon, delta)=({eps:.4f}, {delta})')
        elif method == 'dp_ftrl':
            # DP-FTRL privacy accounting for TreeRestart under fixed order per epoch.
            epochs_between_restarts = [epochs]
            if FLAGS.restart and FLAGS.restart > 0:
                epochs_between_restarts = [FLAGS.restart] * (epochs // FLAGS.restart)
                rem = epochs % FLAGS.restart
                if rem:
                    epochs_between_restarts.append(rem)
            noise = noise_multiplier * clip / batch
            eps = compute_epsilon_tree(num_batches, epochs_between_restarts, noise, delta,
                                       tree_completion=FLAGS.tree_completion, verbose=False)
            print(f'[Privacy] DP-FTRL: (epsilon, delta)=({eps:.4f}, {delta})')
    except Exception as e:
        print(f'[WARN] Failed to report privacy epsilon due to: {e}')


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
