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


from __future__ import annotations

import warnings
from enum import Enum
from typing import Sequence

import brainstate as bst
import brainunit as u

import jax.tree

__all__ = [
    'CompilationError',
    'NotSupportedError',
]


def check_dict_keys(
    d1: dict,
    d2: dict,
):
    """
    Check the keys of two dictionaries.

    Parameters
    ----------
    d1 : dict
      The first dictionary.
    d2 : dict
      The second dictionary.

    Raises
    ------
    ValueError
      If the keys of the two dictionaries are not the same.
    """
    if d1.keys() != d2.keys():
        raise ValueError(f'The keys of the two dictionaries are not the same: {d1.keys()} != {d2.keys()}.')




def _dimensionless(x):
    if isinstance(x, u.Quantity):
        return x.mantissa
    else:
        return x


def remove_units(xs):
    return jax.tree.map(
        _dimensionless,
        xs,
        is_leaf=u.math.is_quantity
    )


git_issue_addr = 'https://github.com/chaobrain/brainscale/issues'


def deprecation_getattr(module, deprecations):
    def getattr(name):
        if name in deprecations:
            message, fn = deprecations[name]
            if fn is None:  # Is the deprecation accelerated?
                raise AttributeError(message)
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return getattr


class NotSupportedError(Exception):
    __module__ = 'brainscale'


class CompilationError(Exception):
    __module__ = 'brainscale'


def state_traceback(states: Sequence[bst.State]):
    """
    Traceback the states of the brain model.

    Parameters
    ----------
    states : Sequence[bst.State]
      The states of the brain model.

    Returns
    -------
    str
      The traceback information of the states.
    """
    state_info = []
    for i, state in enumerate(states):
        state_info.append(
            f'State {i}: {state}\n'
            f'defined at \n'
            f'{state.source_info.traceback}\n'
        )
    return '\n'.join(state_info)


def set_module_as(module: str = 'brainscale'):
    def wrapper(fun: callable):
        fun.__module__ = module
        return fun

    return wrapper


class BaseEnum(Enum):
    @classmethod
    def get_by_name(cls, name: str):
        all_names = []
        for item in cls:
            all_names.append(item.name)
            if item.name == name:
                return item
        raise ValueError(f'Cannot find the {cls.__name__} type {name}. Only support {all_names}.')

    @classmethod
    def get(cls, item: str | Enum):
        if isinstance(item, cls):
            return item
        elif isinstance(item, str):
            return cls.get_by_name(item)
        else:
            raise ValueError(f'Cannot find the {cls.__name__} type {item}.')
