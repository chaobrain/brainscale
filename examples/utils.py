# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os

__all__ = [
    'MyArgumentParser'
]


def _set_gpu_preallocation(mode: float):
    """GPU memory allocation.

    If preallocation is enabled, this makes JAX preallocate ``percent`` of the total GPU memory,
    instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
    """
    assert isinstance(mode, float) and 0. <= mode < 1., f'GPU memory preallocation must be in [0., 1.]. But got {mode}.'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mode)


def _set_gpu_device(device_ids):
    if isinstance(device_ids, int):
        device_ids = str(device_ids)
    elif isinstance(device_ids, (tuple, list)):
        device_ids = ','.join([str(d) for d in device_ids])
    elif isinstance(device_ids, str):
        device_ids = device_ids
    else:
        raise ValueError
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids


class MyArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, gpu_pre_allocate=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
        self.add_argument("--method", type=str, default='bptt', help="Training method.")
        args, _ = self.parse_known_args()

        # device management
        _set_gpu_device(args.devices)
        _set_gpu_preallocation(gpu_pre_allocate)

        # training method
        if args.method != 'bptt':
            self.add_argument(
                "--vjp_time", type=str, default='t', choices=['t', 't_minus_1'],
                help="The VJP time,should be t or t-1."
            )
            if args.method != 'diag':
                self.add_argument(
                    "--etrace_decay", type=float, default=0.9,
                    help="The time constant of eligibility trace "
                )
