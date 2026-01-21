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
import json
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import trange
import numpy as np

import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from optimizers import FTRLOptimizer
from ftrl_noise import CummuNoiseTorch, CummuNoiseEffTorch
from nn import get_nn
from data import get_data
import privacy as privacy_lib
import utils
from utils import EasyDict


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'emnist_merge', ['mnist', 'cifar10', 'emnist_merge'], '')

# Training algorithm.
# - ftrl_dp: DP-FTRL / DP-FTRLM (tree noise; Opacus used for clipping only)
# - ftrl_nodp: vanilla (non-private) FTRL/FTRLM
# - sgd_amp: DP-SGD with Opacus accounting (subsampling amplification)
# - sgd_noamp: DP-SGD training, but privacy is *reported* without subsampling amplification
flags.DEFINE_enum('algo', 'sgd_amp', ['ftrl_dp', 'ftrl_nodp', 'sgd_amp', 'sgd_noamp'],
                  'Training algorithm. See source for details.')

flags.DEFINE_boolean('dp_ftrl', True, 'If True, train with DP-FTRL. If False, train with vanilla FTRL.')
flags.DEFINE_float('noise_multiplier', 0.3, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_integer('restart', 1, 'If > 0, restart the tree every this number of epoch(s).')
flags.DEFINE_boolean('effi_noise', True, 'If True, use tree aggregation proposed in https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf.')
flags.DEFINE_boolean('tree_completion', True, 'If true, generate until reaching a power of 2.')

flags.DEFINE_float('momentum', 0.1, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 0.7, 'Learning rate.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs.')

flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')


flags.DEFINE_integer('run', 1, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')


def main(argv):
    tf.get_logger().setLevel('ERROR')
    tf.config.experimental.set_visible_devices([], "GPU")

    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}") # ADD THIS LINE
    print(f"DEBUG: Detected device = {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}") # ADD THIS LINE
    

    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Data
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    print('Training set size', trainset.image.shape)



    # Hyperparameters for training.
    epochs = FLAGS.epochs
    batch = FLAGS.batch_size if FLAGS.batch_size > 0 else ntrain
    num_batches = ntrain // batch
    noise_multiplier = FLAGS.noise_multiplier
    clip = FLAGS.l2_norm_clip


    # Convert trainset to PyTorch Tensors and create a PyTorch Dataset
    train_images_tensor = torch.Tensor(trainset.image)
    train_labels_tensor = torch.LongTensor(trainset.label)

    

    # Create a PyTorch TensorDataset
    pytorch_train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)

    pytorch_train_loader = torch.utils.data.DataLoader(
    pytorch_train_dataset,
    batch_size=batch, # Use your defined batch size
    shuffle=False,    # DataStream handles shuffling
    drop_last=True    # Ensure consistent batching with DataStream
)


    # Backward-compatible behavior: historically --dp_ftrl toggled private vs non-private FTRL.
    # If the user didn't change --algo from its default, respect --dp_ftrl.
    algo = FLAGS.algo
    if algo in ['ftrl_dp', 'ftrl_nodp'] and (FLAGS.algo == 'ftrl_dp'):
        if not FLAGS.dp_ftrl:
            algo = 'ftrl_nodp'
    dp_ftrl = (algo == 'ftrl_dp')
    dp_sgd = algo in ['sgd_amp', 'sgd_noamp']

    # Determine target delta (matching the paper's common settings).
    def get_delta(dataset_name: str) -> float:
        if dataset_name in ['mnist', 'cifar10']:
            return 1e-5
        return 1e-6

    delta = get_delta(FLAGS.data)

    if not FLAGS.restart:
        FLAGS.tree_completion = False
    lr = FLAGS.learning_rate

    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    #assert report_nimg % batch == 0

    # Get the name of the output directory.
    log_dir = os.path.join(FLAGS.dir, FLAGS.data,
                           utils.get_fn(EasyDict(batch=batch),
                                        EasyDict(dpsgd=(dp_ftrl or dp_sgd), algo=algo,
                                                 restart=FLAGS.restart, completion=FLAGS.tree_completion,
                                                 noise=noise_multiplier, clip=clip, mb=1),
                                        [EasyDict({'lr': lr}),
                                         EasyDict(m=FLAGS.momentum if FLAGS.momentum > 0 else None,
                                                  effi=FLAGS.effi_noise),
                                         EasyDict(sd=FLAGS.run)]
                                        )
                           )
    print('Model dir', log_dir)
    os.makedirs(log_dir, exist_ok=True)

    def _append_jsonl(path: str, obj: dict) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, sort_keys=True) + "\n")

    def _write_json(path: str, obj: dict) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, sort_keys=True)

    # Class to output batches of data
    class DataStream:
        def __init__(self):
            self.shuffle()

        def shuffle(self):
            self.perm = np.random.permutation(ntrain)
            self.i = 0

        def __call__(self):
            if self.i == num_batches:
                self.i = 0
            batch_idx = self.perm[self.i * batch:(self.i + 1) * batch]
            self.i += 1
            return trainset.image[batch_idx], trainset.label[batch_idx]

    data_stream = DataStream()

    # Function to conduct training for one epoch
    def train_loop(model, device, optimizer, cumm_noise, epoch, writer, algo_name: str):
        model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        # Use simple range instead of trange to avoid log file clutter
        loop = range(0, num_batches * batch, batch)
        print(f'Starting Epoch {epoch+1}/{epochs}...')
        step = epoch * num_batches
        for it in loop:
            step += 1
            data, target = data_stream()
            data = torch.Tensor(data).to(device)
            target = torch.LongTensor(target).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # --- REPLACEMENT BLOCK START ---
            if algo_name in ['sgd_amp', 'sgd_noamp']:
                # Standard DP-SGD (Opacus handles everything)
                optimizer.step()
            else:
                # DP-FTRL: We need to bypass Opacus's step to pass our tuple arguments
                
                # 1. Attempt to trigger Opacus privacy logic (Clipping/Accounting)
                #    If these methods don't exist, Opacus might be doing it via hooks (older versions)
                if hasattr(optimizer, 'original_optimizer'): # This will be True for ftrl_dp
                 # Call the original FTRLOptimizer's step method
                    optimizer.original_optimizer.step((lr, cumm_noise()))
                else:
                    # This branch should only be hit for ftrl_nodp (non-DP FTRL)
                    # where Opacus is NOT attached, so 'optimizer' is directly FTRLOptimizer
                    optimizer.step((lr, cumm_noise()))

            losses.append(loss.item())

            # Check if we just finished an epoch (or passed the report interval)
            current_img = step * batch
            if current_img % ntrain ==0:
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, step)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, step)
                model.train()
                print('Step %04d Accuracy %.2f' % (step, 100 * acc_test))

        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print('Epoch %04d Loss %.2f' % (epoch + 1, np.mean(losses)))


    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        b = 1000
        with torch.no_grad():
            accs = [0, 0]
            for i, dataset in enumerate([trainset, testset]):
                for it in trange(0, dataset.image.shape[0], b, leave=False, desc=desc):
                    data, target = dataset.image[it: it + b], dataset.label[it: it + b]
                    data, target = torch.Tensor(data).to(device), torch.LongTensor(target).to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    accs[i] += pred.eq(target.view_as(pred)).sum().item()
                accs[i] /= dataset.image.shape[0]
        return accs

    # Get model for different dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_nn({'mnist': 'small_nn',
                    'emnist_merge': 'small_nn',
                    'cifar10': 'vgg128'}[FLAGS.data],
                   nclass=nclass).to(device)
    

    # Optimizer + privacy engine setup.
    #
    # For DP-FTRL:
    #   - Use Opacus for per-sample gradient computation + clipping only (noise_multiplier=0).
    #   - Inject correlated tree noise via CummuNoise* and update via FTRLOptimizer.
    #
    # For DP-SGD:
    #   - Use standard SGD, and let Opacus add i.i.d. Gaussian noise (noise_multiplier>0).
    #   - We keep the same model, batch size, clipping norm, learning rate, and number of steps
    #     so results are comparable to DP-FTRL.
    privacy_engine = None
    shapes = [p.shape for p in model.parameters()]

    
    if algo in ['ftrl_dp', 'ftrl_nodp']:
        optimizer = FTRLOptimizer(
            model.parameters(),
            momentum=FLAGS.momentum,
            record_last_noise=FLAGS.restart > 0 and FLAGS.tree_completion,
        )
        if dp_ftrl:
            # Initialize PrivacyEngine (simpler constructor)
            privacy_engine = PrivacyEngine() # No sample_rate, etc. here
            privacy_engine.accountant = RDPAccountant()

            # Use make_private to wrap the model and optimizer
            # Note: make_private returns NEW model and optimizer objects
            # You are using Opacus for clipping only (noise_multiplier=0)
            # and manual noise for FTRL, so we pass noise_multiplier=0 to make_private.
            # We also pass batch_size and sample_size instead of sample_rate.
            model, optimizer, _ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=pytorch_train_loader, 
                noise_multiplier=0, # Opacus for clipping only
                max_grad_norm=clip,
                # Pass batch_size and sample_size for Opacus's internal accounting
                batch_size=batch,
                sample_size=ntrain,
                accountant_builder=RDPAccountant
                # alphas=[] is not passed here; Opacus will use its defaults for accounting,
                # but we're not using Opacus's accountant for DP-FTRL anyway.
            )
            print("DEBUG: PrivacyEngine (ftrl_dp) initialized and made private.")
        else:
            # For ftrl_nodp, privacy_engine remains None, as before
            privacy_engine = None # Ensure it's explicitly None if not dp_ftrl
            print("DEBUG: FTRL_NoDP, PrivacyEngine not initialized.")
            
        def get_cumm_noise(effi_noise: bool):
            # When DP is disabled (or noise_multiplier=0), return correctly-shaped zeros so
            # downstream optimizer code never relies on broadcasting.
            if (not dp_ftrl) or noise_multiplier == 0:
                return CummuNoiseTorch(0, shapes, device)
            if not effi_noise:
                return CummuNoiseTorch(noise_multiplier * clip / batch, shapes, device)
            return CummuNoiseEffTorch(noise_multiplier * clip / batch, shapes, device)

        cumm_noise = get_cumm_noise(FLAGS.effi_noise)

    else: # This block handles 'sgd_amp' and 'sgd_noamp'
        # Standard PyTorch SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=FLAGS.momentum)

        # Initialize PrivacyEngine (simpler constructor)
        privacy_engine = PrivacyEngine() # No sample_rate, etc. here
        privacy_engine.accountant = RDPAccountant()

        # Use make_private to wrap the model and optimizer
        # For DP-SGD, noise_multiplier is > 0.
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=pytorch_train_loader, 
            noise_multiplier=noise_multiplier, # This is the actual noise for DP-SGD
            max_grad_norm=clip,
            # Pass batch_size and sample_size for Opacus's internal accounting
            batch_size=batch,
            sample_size=ntrain,
            accountant_builder=RDPAccountant
            # Opacus will use its default RDP orders for accounting
        )
        print("DEBUG: PrivacyEngine (sgd_amp/noamp) initialized and made private.")
        cumm_noise = lambda: [torch.zeros(s, device=device) for s in shapes]

    # The training loop.
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    for epoch in range(epochs):
        if algo in ['sgd_amp', 'sgd_noamp']:
            # DP-SGD baselines use per-epoch reshuffling.
            data_stream.shuffle()
        train_loop(model, device, optimizer, cumm_noise, epoch, writer, algo)

        if epoch + 1 == epochs:
            break
        restart_now = (algo in ['ftrl_dp', 'ftrl_nodp']) and epoch < epochs - 1 and FLAGS.restart > 0 and (epoch + 1) % FLAGS.restart == 0
        if restart_now:
            last_noise = None
            if FLAGS.tree_completion:
                actual_steps = num_batches * FLAGS.restart
                next_pow_2 = 2**(actual_steps - 1).bit_length()
                if next_pow_2 > actual_steps:
                    last_noise = cumm_noise.proceed_until(next_pow_2)
            
            # Unwrap to find the restart method
            if hasattr(optimizer, 'original_optimizer'):
                optimizer.original_optimizer.restart(last_noise)
            elif hasattr(optimizer, 'optimizer'):
                 optimizer.optimizer.restart(last_noise)
            else:
                optimizer.restart(last_noise)
            cumm_noise = get_cumm_noise(FLAGS.effi_noise)
            data_stream.shuffle()  # shuffle the data only when restart

    # Report privacy at the end.
    total_steps = epochs * num_batches
    sample_rate=1.0*batch/ntrain
    eps = None
    if algo == 'ftrl_dp':
        # Translate the restart schedule into the format expected by privacy_lib.
        if FLAGS.restart and FLAGS.restart > 0:
            epochs_between_restarts = [FLAGS.restart] * (epochs // FLAGS.restart)
            if epochs % FLAGS.restart != 0:
                epochs_between_restarts.append(epochs % FLAGS.restart)
        else:
            epochs_between_restarts = [epochs]
        eps = privacy_lib.compute_epsilon_tree(
            num_batches=num_batches,
            epochs_between_restarts=epochs_between_restarts,
            noise=noise_multiplier,
            delta=delta,
            tree_completion=FLAGS.tree_completion,
            verbose=False,
        )
        print(f'Privacy (DP-FTRL): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')
    elif algo == 'sgd_amp':
        # Now get_epsilon is on the accountant object
        eps = privacy_lib.compute_epsilon_sgd_amp(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,  # Use the globally calculated sample_rate
            steps=total_steps,        # Use the globally calculated total_steps
            delta=delta,
            # alphas=None,            # Optional: will use RDPAccountant.DEFAULT_ALPHAS by default
        )
        print(f'Privacy (DP-SGD amp / Custom Accountant): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')
    elif algo == 'sgd_noamp':
        # No-amplification: treat each step as a Gaussian mechanism and compose without subsampling.
        effective_sigma = noise_multiplier / np.sqrt(total_steps)
        eps = privacy_lib.convert_gaussian_renyi_to_dp(effective_sigma, delta, verbose=False)
        print(f'Privacy (DP-SGD no-amp): (ε={eps:.3f}, δ={delta}) over {total_steps} steps')

    # Final evaluation + persistent recording.
    acc_train, acc_test = test(model, device, desc='Final evaluation')
    print(f'Final Accuracy: train={100*acc_train:.2f}%, test={100*acc_test:.2f}%')

    record = {
        "timestamp": datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat(),
        "algo": algo,
        "dataset": FLAGS.data,
        "run": FLAGS.run,
        "ntrain": int(ntrain),
        "ntest": int(testset.image.shape[0]),
        "batch_size": int(batch),
        "epochs": int(epochs),
        "num_batches": int(num_batches),
        "total_steps": int(total_steps),
        "learning_rate": float(lr),
        "momentum": float(FLAGS.momentum),
        "l2_norm_clip": float(clip),
        "noise_multiplier": float(noise_multiplier),
        "restart": int(FLAGS.restart),
        "tree_completion": bool(FLAGS.tree_completion),
        "effi_noise": bool(FLAGS.effi_noise),
        "delta": float(delta),
        "epsilon": (None if eps is None else float(eps)),
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "log_dir": log_dir,
    }

    # Save a per-run summary into the run directory.
    _write_json(os.path.join(log_dir, 'summary.json'), record)
    # Append into a dataset-level and root-level JSONL for easy sweeping/plotting.
    _append_jsonl(os.path.join(FLAGS.dir, FLAGS.data, 'results.jsonl'), record)
    _append_jsonl(os.path.join(FLAGS.dir, 'results.jsonl'), record)

    writer.close()


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
