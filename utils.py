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

"""Utility helpers.

This repo originally used TensorFlow only for TFDS loading and to silence TF logs.
To make the codebase PyTorch-only, we remove TensorFlow dependencies entirely.
"""

import os


def setup_tf():
    """Backward-compatible no-op.

    Older versions of this repo called setup_tf() to suppress TensorFlow logs.
    TensorFlow is no longer used, so this function intentionally does nothing.
    """


class EasyDict(dict):
    """Dictionary with attribute-style access (d.key instead of d['key'])."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_fn(general_params, privacy_params, paramss, find_next=False):
    """Generate the output directory name given parameters."""
    dire = '_'.join([k + str(v) for k, v in general_params.items()]) + '/'

    # We name runs by method rather than a legacy boolean flag.
    # Expected: method in {dp_ftrl, dp_sgd_amp, dp_sgd_noamp}.
    method = getattr(privacy_params, 'method', None)
    if method is None:
        # Backward compat: infer from legacy flags if present.
        if getattr(privacy_params, 'dpsgd', False):
            method = 'dp_sgd'
        else:
            method = 'nonpriv'

    fn = [str(method)]

    # Add key privacy params for easier filtering in TensorBoard.
    # Convert to plain dict for stable iteration.
    try:
        pp_items = dict(privacy_params).items()
    except Exception:
        pp_items = getattr(privacy_params, '__dict__', {}).items()

    skip = {'method', 'dpsgd'}
    parts = []
    for k, v in sorted(pp_items, key=lambda kv: kv[0]):
        if k in skip:
            continue
        if v is None or v is False:
            continue
        if isinstance(v, bool):
            parts.append(k)
        else:
            parts.append(k + str(v))
    if parts:
        fn.append('_'.join(parts))

    for params in paramss:
        s = []
        for k in sorted(params):
            if params[k] is None or params[k] is False:
                continue
            if isinstance(params[k], bool):
                s.append(k)
            else:
                s.append(k + str(params[k]))
        s = '_'.join(s)
        if s is not None and len(s) > 0:
            fn.append(s)

    fn = dire + '_'.join(fn)
    if not find_next:
        return fn

    if not os.path.exists(fn):
        return fn
    i = 1
    while os.path.exists(fn + '_{}'.format(i)):
        i += 1
    return fn + '_{}'.format(i)
