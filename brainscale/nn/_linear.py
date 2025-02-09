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

from typing import Callable, Union, Sequence, Optional

import brainunit as u

import brainstate as bst
from brainscale._etrace_concepts import ETraceParam
from brainscale._etrace_operators import MatMulOp
from brainscale._typing import ArrayLike
from brainstate import functional, init

__all__ = [
    'Linear',
    'ScaledWSLinear',
    'SignedWLinear',
    'SparseLinear',
    'LoRA',
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
        full_etrace: bool = False,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # w_mask
        self.w_mask = init.param(w_mask, self.in_size + self.out_size)

        # weights
        params = dict(weight=init.param(w_init, self.in_size + self.out_size, allow_none=False))
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size, allow_none=False)

        # weight + op
        self.weight_op = param_type(
            params,
            op=MatMulOp(self.w_mask),
            grad='full' if full_etrace else None
        )

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
        full_etrace: bool = False,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        self.in_size = in_size
        self.out_size = out_size

        # weights
        weight = init.param(w_init, [self.in_size[-1], self.out_size[-1]], allow_none=False)
        op = MatMulOp(weight_mask=w_sign, weight_fn=u.math.abs)
        self.weight_op = param_type({'weight': weight}, op=op, grad='full' if full_etrace else None)

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
        full_etrace: bool = False,
        ws_gain: bool = True,
        eps: float = 1e-4,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
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
        self.weight_op = param_type(params, self._operation, grad='full' if full_etrace else None)

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


class SparseLinear(bst.nn.Module):
    """
    Linear layer with Sparse Matrix (can be ``brainunit.sparse.CSR``,
    ``brainunit.sparse.CSC``, ``brainunit.sparse.COO``, or any other sparse matrix).

    Args:
        spar_mat: SparseMatrix. The sparse weight matrix.
        in_size: Size. The input size.
        name: str. The object name.
    """
    __module__ = 'brainscale.nn'

    def __init__(
        self,
        spar_mat: u.sparse.SparseMatrix,
        b_init: Optional[Union[Callable, ArrayLike]] = None,
        in_size: bst.typing.Size = None,
        name: Optional[str] = None,
        param_type: type = ETraceParam,
    ):
        super().__init__(name=name)

        # input and output shape
        if in_size is not None:
            self.in_size = in_size
        self.out_size = spar_mat.shape[-1]
        if in_size is not None:
            assert self.in_size[:-1] == self.out_size[:-1], (
                'The first n-1 dimensions of "in_size" '
                'and "out_size" must be the same.'
            )

        # weights
        assert isinstance(spar_mat, u.sparse.SparseMatrix), '"weight" must be a SparseMatrix.'
        self.spar_mat = spar_mat
        params = dict(weight=spar_mat.data)
        if b_init is not None:
            params['bias'] = init.param(b_init, self.out_size[-1], allow_none=False)
        self.weight_op = param_type(params, self._operation)

    def _operation(self, x, w):
        data = w['weight']
        y = x @ self.spar_mat.with_data(data)
        if 'bias' in w:
            y = y + w['bias']
        return y

    def update(self, x):
        return self.weight_op.execute(x)


class LoRA(bst.nn.Module):
    r"""A standalone LoRA layer.

    $$
        W_\mathrm{L o R A}=W_{\text {orig }}+\frac{\alpha}{r} B A
    $$

    Example usage::

        >>> import brainstate as bst
        >>> import brainscale
        >>> import jax, jax.numpy as jnp
        >>> layer = brainscale.nn.LoRA(3, 2, 4)
        >>> layer.weight_op.value
    {'lora_a': Array([[ 0.25141352, -0.09826107],
            [ 0.2328382 ,  0.38869813],
            [ 0.27069277,  0.7678282 ]], dtype=float32),
     'lora_b': Array([[-0.8372317 ,  0.21012013, -0.52999765, -0.31939325],
            [ 0.64234126, -0.42980042,  1.2549229 , -0.47134295]],      dtype=float32)}
        >>> # Wrap around existing layer
        >>> linear = bst.nn.Linear(3, 4)
        >>> wrapper = brainscale.nn.LoRA(3, 2, 4, base_module=linear)
        >>> assert wrapper.base_module == linear
        >>> y = layer(jnp.ones((16, 3)))
        >>> y.shape
        (16, 4)

    Args:
        in_features: the number of input features.
        lora_rank: the rank of the LoRA dimension.
        out_features: the number of output features.
        base_module: a base module to call and substitute, if possible.
        B_init: initializer function for the weight matrix $B$.
        A_init: initializer function for the weight matrix $A$.
        param_type: the type of the LoRA params.
    """

    def __init__(
        self,
        in_features: bst.typing.Size,
        lora_rank: int,
        out_features: bst.typing.Size,
        *,
        alpha: float = 1.,
        base_module: Optional[bst.nn.Module] = None,
        B_init: Union[Callable, ArrayLike] = init.ZeroInit(),
        A_init: Union[Callable, ArrayLike] = init.LecunNormal(),
        param_type: type = ETraceParam,
    ):
        super().__init__()

        # input and output shape
        self.in_size = in_features
        self.out_size = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.alpha = alpha

        # others
        self.base_module = base_module

        # weights
        param = dict(
            B=B_init((self.in_size[-1], lora_rank)),
            A=A_init((lora_rank, self.out_size[-1]))
        )
        self.weight_op = param_type(param, self._operation)

    def _operation(self, x, w):
        return self.alpha / self.lora_rank * x @ w['B'] @ w['A']

    def __call__(self, x: ArrayLike):
        out = self.weight_op.execute(x)
        if self.base_module is not None:
            if not callable(self.base_module):
                raise ValueError('`self.base_module` must be callable.')
            out += self.base_module(x)
        return out
