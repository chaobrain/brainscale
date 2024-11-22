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

import numbers
from typing import Callable, Union, Sequence, Optional

import brainstate as bst
import brainunit as u
from brainstate import functional, init

from brainscale._etrace_concepts import ETraceParamOp, NonTempParamOp
from brainscale._etrace_operators import MatMulETraceOp
from brainscale._typing import ArrayLike

__all__ = [
    'Linear', 'ScaledWSLinear', 'SignedWLinear', 'CSRLinear',
]


class Linear(bst.nn.Module):
    """
    Linear layer.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        b_init: Optional[Union[Callable, ArrayLike]] = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        as_etrace_weight: bool = True,
        full_etrace: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        self.w_mask = init.param(w_mask, self.in_size + self.out_size)

        # weights
        op = MatMulETraceOp(self.w_mask)
        params = dict(weight=init.param(w_init, self.in_size + self.out_size, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size, allow_none=False)

        # weight + op
        if as_etrace_weight:
            self.weight_op = ETraceParamOp(params, op, grad='full' if full_etrace else None)
        else:
            self.weight_op = NonTempParamOp(params, op.fun)

    def update(self, x):
        return self.weight_op.execute(x)


class SignedWLinear(bst.nn.Module):
    """
    Linear layer with signed weights.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Union[Callable, ArrayLike] = init.KaimingNormal(),
        w_sign: Optional[ArrayLike] = None,
        as_etrace_weight: bool = True,
        full_etrace: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        self.w_sign = w_sign

        # weights
        weight = init.param(w_init, [self.in_size[-1], self.out_size[-1]], allow_none=False)
        if as_etrace_weight:
            self.weight_op = ETraceParamOp(weight, self._operation, grad='full' if full_etrace else None)
        else:
            self.weight_op = NonTempParamOp(weight, self._operation)

    def _operation(self, x, w):
        if self.w_sign is None:
            return u.math.matmul(x, u.math.abs(w))
        else:
            return u.math.matmul(x, u.math.abs(w) * self.w_sign)

    def update(self, x):
        return self.weight_op.execute(x)


class ScaledWSLinear(bst.nn.Module):
    """
    Linear Layer with Weight Standardization.

    Applies weight standardization to the weights of the linear layer.

    Parameters
    ----------
    in_size: int, sequence of int
      The input size.
    out_size: int, sequence of int
      The output size.
    w_init: Callable, ArrayLike
      The initializer for the weights.
    b_init: Callable, ArrayLike
      The initializer for the bias.
    w_mask: ArrayLike, Callable
      The optional mask of the weights.
    as_etrace_weight: bool
      Whether to use ETraceParamOp for the weights.
    ws_gain: bool
      Whether to use gain for the weights. The default is True.
    eps: float
      The epsilon value for the weight standardization.
    name: str
      The name of the object.

    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        in_size: Union[int, Sequence[int]],
        out_size: Union[int, Sequence[int]],
        w_init: Callable = init.KaimingNormal(),
        b_init: Callable = init.ZeroInit(),
        w_mask: Optional[Union[ArrayLike, Callable]] = None,
        as_etrace_weight: bool = True,
        full_etrace: bool = False,
        ws_gain: bool = True,
        eps: float = 1e-4,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        self.w_mask = init.param(w_mask, (self.in_size[0], 1))

        # parameters
        self.eps = eps

        # weights
        params = dict(weight=init.param(w_init, self.in_size + self.out_size, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size, allow_none=False)
        # gain
        if ws_gain:
            s = params['weight'].shape
            params['gain'] = u.math.ones((1,) * (len(s) - 1) + (s[-1],), dtype=params['weight'].dtype)

        # weight operation
        if as_etrace_weight:
            self.weight_op = ETraceParamOp(params, self._operation, grad='full' if full_etrace else None)
        else:
            self.weight_op = NonTempParamOp(params, self._operation)

    def update(self, x):
        return self.weight_op.execute(x)

    def _operation(self, x, params):
        w = params['weight']
        w = functional.weight_standardization(w, self.eps, params.get('gain', None))
        if self.w_mask is not None:
            w = w * self.w_mask
        y = u.math.dot(x, w)
        if 'bias' in params:
            y = y + params['bias']
        return y


class CSRLinear(bst.nn.Module):
    __module__ = 'brainscale.nn'
