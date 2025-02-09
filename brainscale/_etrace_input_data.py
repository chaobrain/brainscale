# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


from typing import Any

import brainstate as bst
import jax
from jax.tree_util import register_pytree_node_class

__all__ = [
    'SingleStepData',
    'MultiStepData',
]


class ETraceData:
    def __init__(self, data: Any):
        self.data = data

    def tree_flatten(self):
        return self.data, ()

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls(data)


@register_pytree_node_class
class SingleStepData(ETraceData):
    """
    The data at a single time step.

    Examples::

        >>> import brainstate as bst
        >>> data = SingleStepData(bst.random.randn(2, 3))

    """
    pass


@register_pytree_node_class
class MultiStepData(ETraceData):
    """
    The data at multiple time steps.

    The first dimension of the data represents the time steps.

    Examples::

        >>> import brainstate as bst
        # data at 10 time steps, each time step has 2 samples, each sample has 3 features
        >>> data = MultiStepData(bst.random.randn(10, 2, 3))


    Another example::

        >>> import brainstate as bst
        >>> data = MultiStepData(
        ...     bst.random.randn(10, 2, 3),
        ...     bst.random.randn(10, 5),
        ... )


    """
    pass


def is_input(x):
    return isinstance(x, (SingleStepData, MultiStepData))


def split_data_types(*args) -> tuple[dict[int, SingleStepData], dict[int, MultiStepData], dict]:
    leaves, tree_def = jax.tree.flatten(args, is_leaf=is_input)
    data_at_single_step = dict()
    data_at_multi_step = dict()
    for i, leaf in enumerate(leaves):
        if isinstance(leaf, SingleStepData):
            data_at_single_step[i] = leaf.data
        elif isinstance(leaf, MultiStepData):
            data_at_multi_step[i] = leaf.data
        else:
            data_at_single_step[i] = leaf

    return data_at_single_step, data_at_multi_step, tree_def


def merge_data(tree_def, *args):
    data = dict()
    for arg in args:
        data.update(arg)
    for i in range(len(data)):
        if i not in data:
            raise ValueError(f"Data at index {i} is missing.")
    return jax.tree.unflatten(tree_def, tuple(data.values()))


def get_single_step_data(*args):
    leaves, tree_def = jax.tree.flatten(args, is_leaf=is_input)
    leaves_processed = []
    for leaf in leaves:
        if isinstance(leaf, SingleStepData):
            leaves_processed.append(leaf.data)
        elif isinstance(leaf, MultiStepData):
            # we need the data at only single time step
            leaves_processed.append(jax.tree.map(lambda x: x[0], leaf.data))
        else:
            leaves_processed.append(leaf)
    args = jax.tree.unflatten(tree_def, leaves_processed)
    return args


def has_multistep_data(*args):
    leaves, _ = jax.tree.flatten(args, is_leaf=is_input)
    return any(isinstance(leaf, MultiStepData) for leaf in leaves)
