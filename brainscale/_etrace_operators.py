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

import contextlib
import threading
from typing import Callable, Optional, Dict

import brainstate as bst
import brainunit as u
import jax
import numpy as np

__all__ = [
    'stop_param_gradients',  # stop weight gradients
    'ETraceOp',  # base class
    'MatMulOp',  # x @ f(w * m) + b
    'ElemWiseOp',  # element-wise operation
]

_etrace_op_name = '_etrace_operator_call'
_etrace_op_name_enable_grad = f'{_etrace_op_name}_enable_grad_'
_etrace_op_name_elemwise = f'{_etrace_op_name}_enable_grad_elemwise_'

X = bst.typing.ArrayLike
W = bst.typing.PyTree
Y = bst.typing.ArrayLike


class OperatorContext(threading.local):
    """
    The context for the eligibility trace operator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_param_gradient = [False]


context = OperatorContext()


@contextlib.contextmanager
def stop_param_gradients(stop_or_not: bool = True):
    """
    Stop the weight gradients for the ETrace weight operator.

    Example::

      >>> import brainscale
      >>> with brainscale.stop_weight_gradients():
      >>>    # do something

    Args:
        stop_or_not: Whether to stop the weight gradients.
    """
    try:
        context.stop_param_gradient.append(stop_or_not)
        yield
    finally:
        context.stop_param_gradient.pop()


def wrap_etrace_fun(fun, name: str = _etrace_op_name):
    fun.__name__ = name
    return fun


def is_etrace_op(jit_param_name: str):
    """
    Check whether the jitted parameter name is the operator.
    """
    return jit_param_name.startswith(_etrace_op_name)


def is_etrace_op_enable_gradient(jit_param_name: str):
    """
    Check whether the jitted parameter name is the operator with the gradient enabled.
    """
    return jit_param_name.startswith(_etrace_op_name_enable_grad)


def is_etrace_op_elemwise(jit_param_name: str):
    """
    Check whether the jitted parameter name is the element-wise operator.
    """
    return jit_param_name.startswith(_etrace_op_name_elemwise)


class ETraceOp:
    """
    The Eligibility Trace Operator.

    The function must have the signature: ``(x: jax.Array, weight: PyTree) -> jax.Array``.

    Attributes:
        fun: The operator function.
        is_diagonal: bool. Whether the operator is in the hidden diagonal or not.

    Args:
        is_diagonal: bool. Whether the operator is in the hidden diagonal or not.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        is_diagonal: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        super().__init__()

        # whether the operator is in the hidden diagonal
        self.is_diagonal = is_diagonal

        # function JIT name
        if name is None:
            name = (
                _etrace_op_name_enable_grad
                if is_diagonal else
                _etrace_op_name
            )

        # JIT the operator function
        # This is important during compilation of eligibility trace graph
        self._jitted_call = jax.jit(wrap_etrace_fun(self._define_call(), name))

    def _define_call(self):
        return lambda x, weights: self.xw_to_y(x, weights)

    def __call__(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        y = self._jitted_call(inputs, weights)
        if context.stop_param_gradient[-1] and not self.is_diagonal:
            y = jax.lax.stop_gradient(y)
        return y

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(is_diagonal={self.is_diagonal})'

    def xw_to_y(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        """
        This function is used to compute the output of the operator.

        It computes:

        $$
        y = f(x, w)
        $$

        Args:
            inputs: The input data.
            weights: The weight parameters.

        Returns:
            The output of the operator.
        """
        raise NotImplementedError

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        """
        This function is used to compute the weight from the hidden dimensional array.

        It computes:

        $$
        w = f(y, w)
        $$

        This function is mainly used when computing eligibility trace updates based on
        :py:class:`ParamDimVjpAlgorithm`.
        """
        raise NotImplementedError

    def xy_to_w(
        self,
        input_dim_arr: X,
        hidden_dim_arr: Y,
    ) -> W:
        """
        This function is used to compute the weight dimensional array from the input and hidden dimensional inputs.

        It computes:

        $$
        w = f(x, y)
        $$

        This function is mainly used when computing eligibility trace updates based on
        :py:class:`IODimVjpAlgorithm`.

        Args:
            input_dim_arr: The input dimensional array.
            hidden_dim_arr: The hidden dimensional array.

        Returns:
            The weight dimensional array.
        """
        raise NotImplementedError


class MatMulOp(ETraceOp):
    """
    The matrix multiplication operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    $$
    y = x @ f(w * m) + b
    $$

    $b$ is the bias term, which can be optional, $m$ is the weight mask,
    and $f$ is the weight function.

    By default, the weight function is the identity function, and
    the weight mask is None.
    """

    def __init__(
        self,
        weight_mask: Optional[jax.Array] = None,
        weight_fn: Callable[[X], X] = lambda w: w,
    ):
        super().__init__(is_diagonal=False)

        # weight mask
        if weight_mask is None:
            pass
        elif isinstance(weight_mask, (np.ndarray, jax.Array, u.Quantity)):
            weight_mask = u.math.asarray(weight_mask)
        else:
            raise TypeError(f'The weight_mask must be an array-like. But got {type(weight_mask)}')
        self.weight_mask = weight_mask

        # weight function
        assert callable(weight_fn), f'The weight_fn must be callable. But got {type(weight_fn)}'
        self.weight_fn = weight_fn

    def xw_to_y(
        self,
        x: bst.typing.ArrayLike,
        w: Dict[str, bst.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

        $$
        y = x @ f(w * m) + b
        $$

        if the bias is provided.

        $$
        y = x @ f(w * m)
        $$

        if the bias is not provided.

        """
        if not isinstance(w, dict):
            raise TypeError(f'{MatMulOp.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')
        weight = w['weight']
        if self.weight_mask is not None:
            weight = weight * self.weight_mask
        weight = self.weight_fn(weight)
        y = u.math.matmul(x, weight)
        if 'bias' in w:
            y = y + w['bias']
        return y

    def yw_to_w(
        self,
        hidden_dim_arr: bst.typing.ArrayLike,
        weight_dim_tree: Dict[str, bst.typing.ArrayLike],
    ) -> Dict[str, bst.typing.ArrayLike]:
        """
        This function is used to compute the weight from the hidden dimensional array.

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        if not isinstance(hidden_dim_arr, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(f'The hidden_dim_arr must be an array-like. But got {type(hidden_dim_arr)}')
        if not isinstance(weight_dim_tree, dict):
            raise TypeError(f'The weight_dim_tree must be a dictionary. But got {type(weight_dim_tree)}')
        if 'weight' not in weight_dim_tree:
            raise ValueError(f'The weight_dim_tree must contain the key "weight".')

        weight_like = weight_dim_tree['weight']
        bias_like = weight_dim_tree.get('bias', None)
        if hidden_dim_arr.ndim == 1:
            assert weight_like.ndim == 2, (
                f'The weight must be a 2D array when hidden_dim_arr is 1D. '
                f'But got the shape {weight_like.shape}'
            )
            if self.weight_mask is None:
                weight_like = weight_like * u.math.expand_dims(hidden_dim_arr, axis=0)
            else:
                weight_like = (
                    weight_like *
                    self.weight_mask *
                    u.math.expand_dims(hidden_dim_arr, axis=0)
                )
            if bias_like is not None:
                assert bias_like.ndim == 1, (
                    f'The bias must be a 1D array when hidden_dim_arr is 1D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        elif hidden_dim_arr.ndim == 2:
            assert weight_like.ndim == 3, (
                f'The weight must be a 3D array when hidden_dim_arr is 2D. '
                f'But got the shape {weight_like.shape}'
            )
            # assume batch size is the first dimension
            if self.weight_mask is None:
                weight_like = weight_like * u.math.expand_dims(hidden_dim_arr, axis=1)
            else:
                weight_like = (
                    weight_like *
                    u.math.expand_dims(self.weight_mask, axis=0) *
                    u.math.expand_dims(hidden_dim_arr, axis=1)
                )
            if bias_like is not None:
                assert bias_like.ndim == 2, (
                    f'The bias must be a 2D array when hidden_dim_arr is 2D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        else:
            raise ValueError(f'The hidden_dim_arr must be a 1D or 2D array. But got the shape {hidden_dim_arr.shape}')
        if bias_like is None:
            return {'weight': weight_like}
        else:
            return {'weight': weight_like, 'bias': bias_like}

    def xy_to_w(
        self,
        input_dim_arr: X,
        hidden_dim_arr: Y,
    ) -> W:
        """
        This function is used to compute the weight dimensional array from the input and hidden dimensional inputs.

        $$
        w = f(x, y)
        $$

        Args:
            input_dim_arr: The input dimensional array.
            hidden_dim_arr: The hidden dimensional array.

        Returns:
            The weight dimensional array.
        """
        raise NotImplementedError


class ElemWiseOp(ETraceOp):
    """
    The element-wise operator for the eligibility trace-based gradient learning.
    
    This interface can be used to define any element-wise operation between weight parameters and hidden states. 

    .. note::

        Different from the :py:class:`StandardOp`, the element-wise operator does not require the input data.
        Its function signature is ``(w: PyTree) -> ndarray``.

        The most important thing is that the element-wise operator must generate the output with
        the same shape as the hidden states.

    Args:
        fn: the element-wise function, which must have the signature: ``(w: PyTree) -> ndarray``.
    """

    def __init__(
        self,
        fn: Callable = lambda w: w,
    ):
        self._raw_fn = fn
        super().__init__(is_diagonal=True, name=_etrace_op_name_elemwise)

    def _define_call(self):
        return lambda weights: self._raw_fn(weights) * 1.0

    def __call__(self, weights: W) -> Y:
        return self._jitted_call(weights)

    def xw_to_y(self, weights: W) -> Y:
        """
        This function is used to compute the output of the element-wise operator.

        It computes:

        $$
        y = f(w)
        $$

        Args:
            weights: The weight parameters.

        Returns:
            The output of the operator.
        """
        return self._raw_fn(weights)

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        """
        This function is used to compute the weight from the hidden dimensional array.

        It computes:

        $$
        w = f(y, w)
        $$

        Args:
            hidden_dim_arr: The hidden dimensional array.
            weight_dim_tree: The weight dimensional tree.

        Returns:
            The updated weight dimensional tree.
        """
        prim, f_vjp = jax.vjp(self._raw_fn, weight_dim_tree)
        assert hidden_dim_arr.shape == prim.shape, (
            f'The shape of the hidden_dim_arr must be the same as the weight_dim_tree. '
            f'Got {hidden_dim_arr.shape} and {prim.shape}'
        )
        return f_vjp(hidden_dim_arr)[0]

    def xy_to_w(
        self,
        input_dim_arr: Optional[X],
        hidden_dim_arr: Y,
    ) -> W:
        """
        This function is used to compute the weight dimensional array from the input and hidden dimensional inputs.

        It computes:

        $$
        w = f(x, y)
        $$

        Args:
            input_dim_arr: The input dimensional array. It is None.
            hidden_dim_arr: The hidden dimensional array.

        Returns:
            The weight dimensional array.
        """
        raise NotImplementedError
