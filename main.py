# main.py

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
import math # Added for math.ceil
from typing import Iterable, List # No longer needed here, moved to dp_data.py

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm # Changed from trange to tqdm for DataLoader iteration
import numpy as np

import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Sampler, TensorDataset # Added for DataLoader
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

from optimizers import FTRLOptimizer
from ftrl_noise import CummuNoiseTorch, CummuNoiseEffTorch
from nn import get_nn
from data import get_data
import privacy as privacy_lib
import utils
from utils import EasyDict

from dp_data import get_dp_dataloader # Import the new function


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'emnist_merge', ['mnist', 'cifar10', 'emnist_merge'], '')
flags.DEFINE_boolean('dp_dataloader', False, 'If True, use differentially private batch sizes.') # NEW FLAG

# Training algorithm.
# - ftrl_dp: DP-FTRL / DP-FTRLM (tree noise; Opacus used for clipping only)
# - ftrl_nodp: vanilla (non-private) FTRL/FTRLM
# - sgd_amp: DP-SGD with Opacus accounting (subsampling amplification)
# - sgd_noamp: DP-SGD training, but privacy is *reported* without subsampling amplification
flags.DEFINE_enum('algo', 'ftrl_nodp', ['ftrl_dp', 'ftrl_nodp', 'sgd_amp', 'sgd_noamp'],
                  'Training algorithm. See source for details.')

flags.DEFINE_boolean('dp_ftrl', True, 'If True, train with DP-FTRL. If False, train with vanilla FTRL.')
flags.DEFINE_float('noise_multiplier', 4, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_integer('restart', 1, 'If > 0, restart the tree every this number of epoch(s).')
flags.DEFINE_boolean('effi_noise', True, 'If True, use tree aggregation proposed in https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf.')
flags.DEFINE_boolean('tree_completion', True, 'If true, generate until reaching a power of 2.')

flags.DEFINE_float('momentum', 0.1, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 0.4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 500, 'Batch size.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs.')

flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')


flags.DEFINE_integer('run', 1, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')


def main(argv):
    tf.get_logger().setLevel('ERROR')
    tf.config.experimental.set_visible_devices([], "GPU")

    print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"DEBUG: Detected device = {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Data
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    print('Training set size', trainset.image.shape)

    # Hyperparameters for training.
    epochs = FLAGS.epochs
    # 'mean_batch_size' is now used consistently
    mean_batch_size = FLAGS.batch_size if FLAGS.batch_size > 0 else ntrain 
    # num_batches_nominal will be an approximation due to variable batch sizes
    num_batches_nominal = math.ceil(ntrain / mean_batch_size) 
    noise_multiplier = FLAGS.noise_multiplier
    clip = FLAGS.l2_norm_clip

    # Convert trainset to PyTorch Tensors and create a PyTorch Dataset
    train_images_tensor = torch.Tensor(trainset.image)
    train_labels_tensor = torch.LongTensor(trainset.label)
    
    # Create a PyTorch TensorDataset (used by both DP and non-DP DataLoader)
    pytorch_train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)

    # Determine target delta (matching the paper's common settings).
    def get_delta(dataset_name: str) -> float:
        if dataset_name in ['mnist', 'cifar10']:
            return 1e-5
        return 1e-6

    delta = get_delta(FLAGS.data)

    # --- Conditional DataLoader Setup ---
    sampler_epsilon = 0.0
    sampler_delta = 0.0
    if FLAGS.dp_dataloader:
        pytorch_train_loader, sampler_epsilon, sampler_delta = get_dp_dataloader(
            trainset_images=train_images_tensor,
            trainset_labels=train_labels_tensor,
            ntrain=ntrain,
            mean_batch_size=mean_batch_size, # Use mean_batch_size
            delta=delta,
            shuffle=True
        )
        print(f"DEBUG: Using DP DataLoader. Sampler Epsilon: {sampler_epsilon:.3f}, Sampler Delta: {sampler_delta}")
    else:
        # Standard DataLoader for non-DP batching
        pytorch_train_loader = DataLoader(
            pytorch_train_dataset,
            batch_size=mean_batch_size, # Fixed batch size
            shuffle=True,     # Standard shuffling
            drop_last=True    # Drop last incomplete batch
        )
        # Sampler epsilon and delta are 0 for non-DP DataLoader
        print(f"DEBUG: Using Non-DP DataLoader (fixed batch size).")
    # --- END Conditional DataLoader Setup ---


    # Backward-compatible behavior: historically --dp_ftrl toggled private vs non-private FTRL.
    # If the user didn't change --algo from its default, respect --dp_ftrl.
    algo = FLAGS.algo
    if algo in ['ftrl_dp', 'ftrl_nodp'] and (FLAGS.algo == 'ftrl_dp'):
        if not FLAGS.dp_ftrl:
            algo = 'ftrl_nodp'
    dp_ftrl = (algo == 'ftrl_dp')
    dp_sgd = algo in ['sgd_amp', 'sgd_noamp']


    if not FLAGS.restart:
        FLAGS.tree_completion = False
    lr = FLAGS.learning_rate

    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    #assert report_nimg % batch == 0 # This assertion might fail with variable batch sizes

    # Get the name of the output directory.
    log_dir = os.path.join(FLAGS.dir, FLAGS.data,
                           utils.get_fn(EasyDict(batch=mean_batch_size), # 'batch' here is mean_batch_size
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

    # Function to conduct training for one epoch
    # MODIFIED: Removed cumm_noise_fn from arguments, as it's now a global object 'cumm_noise'
    def train_loop(model, device, optimizer, epoch, writer, algo_name: str): 
        model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        
        # step_counter now tracks actual batches processed
        step_counter = epoch * len(pytorch_train_loader) 
        
        loop_iterator = tqdm(pytorch_train_loader, leave=False, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx_in_epoch, (data, target) in enumerate(loop_iterator):
            step_counter += 1 # Total steps across all epochs

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # --- MODIFIED REPLACEMENT BLOCK START (for variable batch sizes) ---
            if algo_name in ['sgd_amp', 'sgd_noamp']:
                # Get the actual batch size from the current data batch
                # This is still needed for Opacus's internal clipping/noise scaling
                actual_batch_size = data.shape[0] 
                actual_sample_rate = actual_batch_size / ntrain # Still useful for debugging/logging

                # Opacus's DPOptimizer.step() handles clipping and noise addition
                # correctly for the ACTUAL variable batch size.
                optimizer.step() 

                # Manually update the RDPAccountant with the MEAN sample rate
                # as per the simplification request.
                manual_rdp_accountant.step(
                    noise_multiplier=noise_multiplier, # Use the global noise_multiplier
                    sample_rate=mean_batch_size / ntrain, # <--- FIXED TO MEAN_BATCH_SIZE
                    # steps=1 is default, which is correct for each batch step
                )

            else: # FTRL logic
                # For FTRL, noise is now scaled by mean_batch_size (fixed)
                # as per the simplification request.
                if hasattr(optimizer, 'original_optimizer'): 
                    optimizer.original_optimizer.step((lr, cumm_noise())) # <--- NO ARGUMENT TO CUMM_NOISE
                else:
                    optimizer.step((lr, cumm_noise())) # <--- NO ARGUMENT TO CUMM_NOISE
            # --- END MODIFIED REPLACEMENT BLOCK ---

            losses.append(loss.item())
            
            # Check if we should report accuracy (adjust logic for variable batch sizes)
            # Report every 20% of the nominal batches per epoch, or at least once.
            report_interval_batches = max(1, len(pytorch_train_loader) // 5)
            if (batch_idx_in_epoch + 1) % report_interval_batches == 0 or (batch_idx_in_epoch + 1) == len(pytorch_train_loader):
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, step_counter)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, step_counter)
                model.train()
                print(f'Step {step_counter:04d} Accuracy {100 * acc_test:.2f}%')

        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print(f'Epoch {epoch+1:04d} Loss {np.mean(losses):.2f}')


    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        b = 1000
        with torch.no_grad():
            accs = [0, 0]
            # Use original trainset/testset for evaluation, not DataLoader
            for i, dataset in enumerate([trainset, testset]): 
                for it in tqdm(range(0, dataset.image.shape[0], b), leave=False, desc=desc):
                    data, target = dataset.image[it: it + b], dataset.label[it: it + b]
                    data = torch.Tensor(data).to(device) # Ensure data is tensor before moving to device
                    target = torch.LongTensor(target).to(device)
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
    privacy_engine = None
    shapes = [p.shape for p in model.parameters()]

    # --- NEW: Manual RDP Accountant for variable batch sizes ---
    # This will be updated manually in the train_loop for sgd_amp/noamp
    manual_rdp_accountant = RDPAccountant()
    # --- END NEW ---

    if algo in ['ftrl_dp', 'ftrl_nodp']:
        optimizer = FTRLOptimizer(
            model.parameters(),
            momentum=FLAGS.momentum,
            record_last_noise=FLAGS.restart > 0 and FLAGS.tree_completion, # Kept as per user's original
        )
        if dp_ftrl:
            privacy_engine = PrivacyEngine() # No arguments
            model, optimizer, _ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=pytorch_train_loader, # Pass the DataLoader
                noise_multiplier=0, # Opacus for clipping only
                max_grad_norm=clip,
                batch_size=mean_batch_size, # Pass mean_batch_size for Opacus's internal setup
                sample_size=ntrain,
            )
            print("DEBUG: PrivacyEngine (ftrl_dp) initialized and made private.")
        else:
            privacy_engine = None # Ensure it's explicitly None if not dp_ftrl
            print("DEBUG: FTRL_NoDP, PrivacyEngine not initialized.")
            
        # --- MODIFIED: cumm_noise is now an object, initialized with noise_multiplier and clip ---
        # Fixed std based on mean_batch_size
        if (not dp_ftrl) or noise_multiplier == 0:
            cumm_noise = CummuNoiseTorch(0, shapes, device) # std=0
        elif not FLAGS.effi_noise:
            cumm_noise = CummuNoiseTorch(noise_multiplier * clip / mean_batch_size, shapes, device) # <--- FIXED STD
        else:
            cumm_noise = CummuNoiseEffTorch(noise_multiplier * clip / mean_batch_size, shapes, device) # <--- FIXED STD
        # --- END MODIFIED ---

    else: # This block handles 'sgd_amp' and 'sgd_noamp'
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=FLAGS.momentum)
        privacy_engine = PrivacyEngine() # No arguments
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=pytorch_train_loader, # Pass the DataLoader
            noise_multiplier=noise_multiplier, # This is the actual noise for DP-SGD
            max_grad_norm=clip,
            batch_size=mean_batch_size, # Pass mean_batch_size for Opacus's internal setup
            sample_size=ntrain,
        )
        print("DEBUG: PrivacyEngine (sgd_amp/noamp) initialized and made private.")
        # cumm_noise is not used for SGD, but defined for consistency if needed
        cumm_noise = CummuNoiseTorch(0, shapes, device) # Dummy object with std=0


    # The training loop.
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    total_batches_processed = 0 # Track actual batches for final accounting
    for epoch in range(epochs):
        # MODIFIED: train_loop no longer needs cumm_noise_fn argument
        train_loop(model, device, optimizer, epoch, writer, algo) 
        total_batches_processed += len(pytorch_train_loader) # Add batches from this epoch

        if epoch + 1 == epochs:
            break
        
        # --- MODIFIED: Restart logic for variable batch sizes ---
        restart_now = (algo in ['ftrl_dp', 'ftrl_nodp']) and epoch < epochs - 1 and FLAGS.restart > 0 and (epoch + 1) % FLAGS.restart == 0
        if restart_now:
            print("Performing Tree Restart / Completion... (This may take a moment)")
            last_noise = None
            if FLAGS.tree_completion:
                # num_batches here is nominal, but compute_epsilon_tree uses it for tree structure
                actual_steps_since_last_restart = len(pytorch_train_loader) * FLAGS.restart 
                next_pow_2 = 2**(actual_steps_since_last_restart - 1).bit_length()
                if next_pow_2 > actual_steps_since_last_restart:
                    # For FTRL, noise generation during completion uses fixed std
                    last_noise = cumm_noise.proceed_until(next_pow_2) # <--- NO ARGUMENT
            
            # Unwrap to find the restart method
            if hasattr(optimizer, 'original_optimizer'):
                optimizer.original_optimizer.restart(last_noise)
            elif hasattr(optimizer, 'optimizer'):
                 optimizer.optimizer.restart(last_noise)
            else:
                optimizer.restart(last_noise)

            if (not dp_ftrl) or noise_multiplier == 0:
                cumm_noise = CummuNoiseTorch(0, shapes, device) # std=0
            elif not FLAGS.effi_noise:
                cumm_noise = CummuNoiseTorch(noise_multiplier * clip / mean_batch_size, shapes, device) # <--- FIXED STD
            else:
                cumm_noise = CummuNoiseEffTorch(noise_multiplier * clip / mean_batch_size, shapes, device) # <--- FIXED STD
            # MODIFIED: cumm_noise object is stateful, no need to re-assign
            print("Restart Complete. Resuming training...")
        # --- END MODIFIED ---

    # Report privacy at the end.
    # total_steps for accounting should be based on actual batches processed
    total_steps_for_accounting = total_batches_processed 
    eps = None
    
    final_dp_epsilon = 0.0 # Epsilon from DP-SGD/DP-FTRL mechanism
    
    if algo == 'ftrl_dp':
        if FLAGS.restart and FLAGS.restart > 0:
            epochs_between_restarts = [FLAGS.restart] * (epochs // FLAGS.restart)
            if epochs % FLAGS.restart != 0:
                epochs_between_restarts.append(epochs % FLAGS.restart)
        else:
            epochs_between_restarts = [epochs]
        
        # num_batches here is nominal, but compute_epsilon_tree uses it for tree structure
        # It should ideally be len(pytorch_train_loader) for each restart segment
        eps_ftrl = privacy_lib.compute_epsilon_tree(
            num_batches=len(pytorch_train_loader), # Use actual number of batches per epoch
            epochs_between_restarts=epochs_between_restarts,
            noise=noise_multiplier,
            delta=delta,
            tree_completion=FLAGS.tree_completion,
            verbose=False,
        )
        final_dp_epsilon = eps_ftrl
        print(f'Privacy (DP-FTRL): (ε={eps_ftrl:.3f}, δ={delta}) over {total_steps_for_accounting} steps')

    elif algo == 'sgd_amp':
        # Get epsilon from the manually updated RDP accountant
        eps_dp_sgd = manual_rdp_accountant.get_epsilon(delta)
        final_dp_epsilon = eps_dp_sgd
        print(f'Privacy (DP-SGD amp): (ε={eps_dp_sgd:.3f}, δ={delta}) over {total_steps_for_accounting} steps')
        
    elif algo == 'sgd_noamp':
        # No-amplification: treat each step as a Gaussian mechanism and compose without subsampling.
        effective_sigma = noise_multiplier / np.sqrt(total_steps_for_accounting)
        eps_noamp = privacy_lib.convert_gaussian_renyi_to_dp(effective_sigma, delta, verbose=False)
        final_dp_epsilon = eps_noamp
        print(f'Privacy (DP-SGD no-amp): (ε={eps_noamp:.3f}, δ={delta}) over {total_steps_for_accounting} steps')

    # --- MODIFIED: Separate privacy recording ---
    # The 'eps' variable will now just hold the training privacy
    eps = final_dp_epsilon 
    print(f'Privacy (Training Mechanism): (ε={final_dp_epsilon:.3f}, δ={delta})')
    print(f'Privacy (DataLoader Sampler): (ε={sampler_epsilon:.3f}, δ={sampler_delta})')
    print(f'Privacy (Composed Total): (ε={final_dp_epsilon + sampler_epsilon:.3f}, δ={delta})') # Still print composed

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
        "batch_size": int(mean_batch_size), # This is now mean_batch_size
        "epochs": int(epochs),
        "num_batches": int(total_batches_processed), # Use actual total batches
        "total_steps": int(total_batches_processed),
        "learning_rate": float(lr),
        "momentum": float(FLAGS.momentum),
        "l2_norm_clip": float(clip),
        "noise_multiplier": float(noise_multiplier),
        "restart": int(FLAGS.restart),
        "tree_completion": bool(FLAGS.tree_completion),
        "effi_noise": bool(FLAGS.effi_noise),
        "delta": float(delta),
        "epsilon": (None if eps is None else float(eps)), # This now holds final_dp_epsilon
        "epsilon_dataloader": float(sampler_epsilon), # NEW RECORD
        "delta_dataloader": float(sampler_delta),    # NEW RECORD
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