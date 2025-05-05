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

import os

import numpy as np
import tonic
from tonic.collation import PadTensors
from tonic.datasets import DVSGesture
from torch.utils.data import DataLoader
from tqdm import tqdm


def _dvs_gesture_preprocessing(
    num_workers: int,
    n_step: int = 100,
    cache_dir: str = '/mnt/d/data/gesture/'
):
    """
    Preprocesses the DVS gesture dataset by converting event data into frame-based representation.

    This function loads the DVS Gesture dataset, applies transformations to convert event streams
    into frames, and saves the processed data as NPZ files. If the preprocessed files already
    exist at the specified location, the function returns without further processing.

    Parameters
    ----------
    num_workers : int
        Number of worker processes for data loading.
    n_step : int, default=100
        Number of time bins to divide the event stream into, controlling temporal resolution.
    cache_dir : str, default='/mnt/d/data/gesture/'
        Directory path where the dataset will be downloaded and processed data will be saved.

    Returns
    -------
    None
        The function saves the preprocessed data to NPZ files and doesn't return any value.

    Notes
    -----
    The processed data is saved as two NPZ files (train and test) containing:
    - xs: Sparse event indices
    - ys: Class labels
    - img_size: Shape information for reconstruction
    """
    train_filename = os.path.join(cache_dir, f'DVSGesture/DVSGesture-mlp-train-step={n_step}.npz')
    test_filename = os.path.join(cache_dir, f'DVSGesture/DVSGesture-mlp-test-step={n_step}.npz')
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        return

    batch_size = 128
    in_shape = DVSGesture.sensor_size
    transform = tonic.transforms.Compose(
        [
            tonic.transforms.ToFrame(sensor_size=in_shape, n_time_bins=n_step),
            # transforms.Downsample(time_factor=0.5),
            # transforms.DropEvent(p=0.001),
        ]
    )
    train_set = DVSGesture(save_to=cache_dir, train=True, transform=transform)
    test_set = DVSGesture(save_to=cache_dir, train=False, transform=transform)
    train_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=True),
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=PadTensors(batch_first=True),
        num_workers=num_workers
    )

    def _to_index(xs):
        xs = np.asarray(xs)
        res = []
        for x in xs:
            indices = np.where(np.reshape(x, (x.shape[0], -1)))  # a tuple of two indices
            res.append(np.asarray(indices, dtype=np.int32).T)
        return res

    for loader, filename in [
        (train_loader, train_filename),
        (test_loader, test_filename)
    ]:
        xs, ys = [], []
        img_size = None
        for i, (x, y) in tqdm(enumerate(loader), total=len(loader), desc='preprocessing'):
            xs.extend(_to_index(x))
            ys.append(np.asarray(y))
            if img_size is None:
                img_size = (x.shape[1], np.prod(x.shape[2:]))
        xs = np.asarray(xs, dtype=object)
        ys = np.concatenate(ys)
        np.savez(filename, xs=xs, ys=ys, img_size=np.asarray(img_size, dtype=np.int32))


if __name__ == '__main__':
    for n_seq in [50, 100, 200, 300, 400, 600, 800, 1000]:
        print(f'Processing the data with length of {n_seq}')
        _dvs_gesture_preprocessing(10, n_seq)
