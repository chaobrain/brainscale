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
    'SpMVOp',  # x @ f(sparse_weight) + b
    'LoraOp',  # low-rank approximation
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


class ETraceOp(bst.util.PrettyReprTree):
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

    def __pretty_repr_item__(self, k, v):
        if k == '_jitted_call':
            return None, None
        return k, v

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
        weights: W,
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
            weights: The weight dimensional array.

        Returns:
            The weight dimensional array.
        """
        primals, f_vjp = jax.vjp(
            lambda w: u.get_mantissa(self.xw_to_y(input_dim_arr, w)),  # dimensionless processing
            weights
        )
        assert hidden_dim_arr.shape == primals.shape, (
            f'The shape of the hidden_dim_arr must be the same as the primals. '
            f'Got {hidden_dim_arr.shape} and {primals.shape}'
        )
        return f_vjp(
            # dimensionless processing
            u.get_mantissa(hidden_dim_arr)
        )[0]


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
            raise TypeError(f'{self.__class__.__name__} only supports '
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
        if 'bias' in weight_dim_tree:
            bias_like = weight_dim_tree['bias']
        else:
            bias_like = None
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


class SpMVOp(ETraceOp):
    """
    The sparse matrix-vector multiplication operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    $$
    y = v @ f(w) + b,
    $$

    $b$ is the bias term, which can be optional, $f$ is the weight function, and $v$ is the input vector.

    By default, the weight function is the identity function.

    .. note::

       The sparse matrix must be the instance of ``brainunit.sparse.SparseMatrix``,
       which implements the protocol method ``.yw_to_w()`` that we need.

    """

    def __init__(
        self,
        sparse_mat: u.sparse.SparseMatrix,
        weight_fn: Callable[[X], X] = lambda w: w,
    ):
        super().__init__(is_diagonal=False)

        # sparse matrix
        assert isinstance(sparse_mat, u.sparse.SparseMatrix), (
            f'The sparse_mat must be a SparseMatrix. But we got {type(sparse_mat)}'
        )
        self.sparse_mat = sparse_mat

        # weight function
        assert callable(weight_fn), f'The weight_fn must be callable. But got {type(weight_fn)}'
        self.weight_fn = weight_fn

    def _check_weight(self, w: W, check_shape: bool = True):
        if not isinstance(w, dict):
            raise TypeError(f'{self.__class__.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')
        if check_shape:
            if w['weight'].shape != self.sparse_mat.data.shape:
                raise ValueError(f'The shape of the weight must be the same as the sparse matrix data. '
                                 f'Got {w["weight"].shape} and {self.sparse_mat.data.shape}.')
        if w['weight'].dtype != self.sparse_mat.data.dtype:
            raise ValueError(f'The dtype of the weight must be the same as the sparse matrix data. '
                             f'Got {w["weight"].dtype} and {self.sparse_mat.data.dtype}.')
        if u.get_unit(w['weight']) != u.get_unit(self.sparse_mat.data):
            raise ValueError(f'The unit of the weight must be the same as the sparse matrix data. '
                             f'Got {u.get_unit(w["weight"])} and {u.get_unit(self.sparse_mat.data)}.')

    def xw_to_y(
        self,
        x: bst.typing.ArrayLike,
        w: Dict[str, bst.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

        $$
        y = x @ f(w) + b
        $$
        """
        self._check_weight(w)
        weight = self.weight_fn(w['weight'])
        sparse_mat = self.sparse_mat.with_data(weight)
        y = x @ sparse_mat
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
        self._check_weight(weight_dim_tree, check_shape=False)
        weight_like: bst.typing.ArrayLike = weight_dim_tree['weight']
        if 'bias' in weight_dim_tree:
            bias_like = weight_dim_tree['bias']
        else:
            bias_like = None
        assert hidden_dim_arr.ndim == 1, (
            f'The hidden_dim_arr must be a 1D array. But got the shape {hidden_dim_arr.shape}'
        )
        weight_like = self.sparse_mat.yw_to_w(
            u.math.expand_dims(hidden_dim_arr, axis=0),
            weight_like
        )
        if bias_like is not None:
            assert bias_like.ndim == 1, (
                f'The bias must be a 1D array when hidden_dim_arr is 1D. '
                f'But got the shape {bias_like.shape}'
            )
            bias_like = bias_like * hidden_dim_arr
        if bias_like is None:
            return {'weight': weight_like}
        else:
            return {'weight': weight_like, 'bias': bias_like}


class LoraOp(ETraceOp):
    r"""
    The low-rank approximation operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

    $$
    y = \alpha x B A + b
    $$

    $b$ is the bias term, which can be optional, $\alpha$ is the scaling factor,
    $A$ is the weight matrix, $B$ is the low-rank matrix, and $x$ is the input data.

    """

    def __init__(
        self,
        alpha: Optional[bst.typing.ArrayLike] = None,
    ):
        super().__init__(is_diagonal=False)

        # weight mask
        if alpha is not None:
            alpha = u.math.asarray(alpha)
        self.alpha = alpha

    def _check_weight(self, w: W):
        if not isinstance(w, dict):
            raise TypeError(f'{self.__class__.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'B' not in w:
            raise ValueError(f'The weight must contain the key "B".')
        if 'A' not in w:
            raise ValueError(f'The weight must contain the key "A".')

    def xw_to_y(
        self,
        x: bst.typing.ArrayLike,
        w: Dict[str, bst.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

        $$
        y = \alpha * x @ B @ A + b
        $$

        Args:
            x: The input data.
            w: The weight parameters.

        Returns:
            The output of the operator.
        """
        self._check_weight(w)
        if self.alpha is not None:
            x = self.alpha * x
        y = x @ w['B'] @ w['A']
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
        self._check_weight(weight_dim_tree)

        B_like = weight_dim_tree['B']
        A_like = weight_dim_tree['A']
        if 'bias' in weight_dim_tree:
            bias_like = weight_dim_tree['bias']
        else:
            bias_like = None
        if hidden_dim_arr.ndim == 1:
            assert B_like.ndim == 2 and A_like.ndim == 2, (
                f'The weight must be a 2D array when hidden_dim_arr is 1D. '
                f'But got the shape of B = {B_like.shape}, A = {A_like.shape}.'
            )
            A_like = (
                A_like *
                u.math.expand_dims(hidden_dim_arr, axis=0)
            )
            if bias_like is not None:
                assert bias_like.ndim == 1, (
                    f'The bias must be a 1D array when hidden_dim_arr is 1D. '
                    f'But got the shape {bias_like.shape}'
                )
                bias_like = bias_like * hidden_dim_arr
        elif hidden_dim_arr.ndim == 2:
            assert B_like.ndim == 3 and A_like.ndim == 3, (
                f'The weight must be a 3D array when hidden_dim_arr is 2D. '
                f'But got the shape B = {B_like.shape}, A = {A_like.shape}.'
            )
            # assume batch size is the first dimension
            A_like = (
                A_like *
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
            return {'B': B_like, 'A': A_like}
        else:
            return {'B': B_like, 'A': A_like, 'bias': bias_like}


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

    def __pretty_repr_item__(self, k, v):
        if k in ['_raw_fn', '_jitted_call']:
            return None, None
        return k, v

    def _define_call(self):
        return lambda weights: self._raw_fn(weights) * 1.0

    def __call__(self, weights: W) -> Y:
        return self._jitted_call(weights)

    def xw_to_y(
        self,
        inputs: Optional[X],
        weights: W
    ) -> Y:
        """
        This function is used to compute the output of the element-wise operator.

        It computes:

        $$
        y = f(w)
        $$

        Args:
            inputs: The input data. It is None.
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
        prim, f_vjp = jax.vjp(
            # dimensionless processing
            lambda w: u.get_mantissa(self._raw_fn(w)),
            weight_dim_tree
        )
        assert hidden_dim_arr.shape == prim.shape, (
            f'The shape of the hidden_dim_arr must be the same as the weight_dim_tree. '
            f'Got {hidden_dim_arr.shape} and {prim.shape}'
        )
        return f_vjp(
            # dimensionless processing
            u.get_mantissa(hidden_dim_arr)
        )[0]

    def xy_to_w(
        self,
        input_dim_arr: Optional[X],
        hidden_dim_arr: Y,
        weights: W,
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
            weights: The weight dimensional

        Returns:
            The weight dimensional array.
        """

        primals, f_vjp = jax.vjp(
            # dimensionless processing
            lambda w: u.get_mantissa(self._raw_fn(w)),
            weights
        )
        assert hidden_dim_arr.shape == primals.shape, (
            f'The shape of the hidden_dim_arr must be the same as the primals. '
            f'Got {hidden_dim_arr.shape} and {primals.shape}'
        )
        return f_vjp(
            # dimensionless processing
            u.get_mantissa(hidden_dim_arr)
        )[0]
