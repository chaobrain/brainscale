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

# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import partial
from typing import Callable, Union, Sequence, Optional, Any

import brainstate as bst
from brainscale._etrace_concepts import ETraceParam, NonTempParam, ETraceState
from brainscale._typing import ArrayLike, Size, Axes
from brainstate.nn._interaction._normalizations import _BatchNorm

__all__ = [
    'BatchNorm0d', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'LayerNorm', 'RMSNorm', 'GroupNorm'
]


def _operation(x, param):
    if 'scale' in param:
        x = x * param['scale']
    if 'bias' in param:
        x = x + param['bias']
    return x


class _BatchNormETrace(_BatchNorm):
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Size,
        feature_axis: Axes = -1,
        track_running_stats: bool = True,
        epsilon: float = 1e-5,
        momentum: float = 0.99,
        affine: bool = True,
        bias_initializer: Union[ArrayLike, Callable] = bst.init.Constant(0.),
        scale_initializer: Union[ArrayLike, Callable] = bst.init.Constant(1.),
        axis_name: Optional[Union[str, Sequence[str]]] = None,
        axis_index_groups: Optional[Sequence[Sequence[int]]] = None,
        as_etrace_weight: bool = False,
        full_etrace: bool = False,
        name: Optional[str] = None,
        dtype: Any = None,
        mean_type: type = ETraceState
    ):

        if as_etrace_weight:
            weight_type = partial(
                ETraceParam,
                op=_operation,
                grad='full' if full_etrace else None,
                is_diagonal=True
            )
        else:
            weight_type = partial(NonTempParam, op=_operation)

        super().__init__(
            in_size=in_size,
            feature_axis=feature_axis,
            track_running_stats=track_running_stats,
            epsilon=epsilon,
            momentum=momentum,
            affine=affine,
            bias_initializer=bias_initializer,
            scale_initializer=scale_initializer,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups,
            dtype=dtype,
            mean_type=mean_type,
            param_type=weight_type,
            name=name
        )


class BatchNorm0d(_BatchNormETrace):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.BatchNorm0d.__doc__
    num_spatial_dims: int = 0


class BatchNorm1d(_BatchNormETrace):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.BatchNorm1d.__doc__
    num_spatial_dims: int = 1


class BatchNorm2d(_BatchNormETrace):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.BatchNorm2d.__doc__
    num_spatial_dims: int = 2


class BatchNorm3d(_BatchNormETrace):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.BatchNorm3d.__doc__
    num_spatial_dims: int = 3


class LayerNorm(bst.nn.LayerNorm):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.LayerNorm.__doc__

    def __init__(
        self,
        *args,
        as_etrace_weight: bool = True,
        full_etrace: bool = False,
        **kwargs,
    ):
        if as_etrace_weight:
            weight_type = partial(
                ETraceParam,
                op=_operation,
                grad='full' if full_etrace else None,
                is_diagonal=True
            )
        else:
            weight_type = partial(NonTempParam, op=_operation)
        super().__init__(*args, param_type=weight_type, **kwargs)


class RMSNorm(bst.nn.LayerNorm):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.RMSNorm.__doc__

    def __init__(
        self,
        *args,
        as_etrace_weight: bool = True,
        full_etrace: bool = False,
        **kwargs,
    ):
        if as_etrace_weight:
            weight_type = partial(
                ETraceParam,
                op=_operation,
                grad='full' if full_etrace else None,
                is_diagonal=True
            )
        else:
            weight_type = partial(NonTempParam, op=_operation)
        super().__init__(*args, param_type=weight_type, **kwargs)


class GroupNorm(bst.nn.LayerNorm):
    __module__ = 'brainscale.nn'
    __doc__ = bst.nn.GroupNorm.__doc__

    def __init__(
        self,
        *args,
        as_etrace_weight: bool = False,
        full_etrace: bool = False,
        **kwargs,
    ):
        if as_etrace_weight:
            weight_type = partial(
                ETraceParam,
                op=_operation,
                grad='full' if full_etrace else None,
                is_diagonal=True
            )
        else:
            weight_type = partial(NonTempParam, op=_operation)
        super().__init__(*args, param_type=weight_type, **kwargs)
