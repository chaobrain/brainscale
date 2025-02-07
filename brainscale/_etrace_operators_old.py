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
from functools import reduce, partial
from typing import Callable, Optional, Dict, List, Tuple

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from ._misc import remove_units

__all__ = [
    'stop_param_gradients',  # stop weight gradients
    'ETraceOp',  # base class
    'MatmulOp',  # x @ w + b
    'AbsMatmulOp',  # x @ |w| + b
    'ElemWiseOp',  # element-wise operation
]

_etrace_op_name = '_etrace_weight_operator_call'
_etrace_op_name_enable_grad = f'{_etrace_op_name}_enable_grad_'
_etrace_op_name_enable_grad_elem = f'{_etrace_op_name}_enable_grad_elemwise_'

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
    return jit_param_name.startswith(_etrace_op_name)


def is_etrace_op_enable_gradient(jit_param_name: str):
    return jit_param_name.startswith(_etrace_op_name_enable_grad)


class ETraceOp:
    """
    The Eligibility Trace Operator.

    The function must have the signature: ``(x: jax.Array, weight: PyTree) -> jax.Array``.

    Attributes:
        fun: The operator function.
        is_diagonal: bool. Whether the operator is in the hidden diagonal or not.

    Args:
        fun: The operator function.
        is_diagonal: bool. Whether the operator is in the hidden diagonal or not.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        fun: Callable,
        is_diagonal: bool = False
    ):
        super().__init__()
        self.fun = fun
        self.is_diagonal = is_diagonal
        name = (
            _etrace_op_name_enable_grad
            if is_diagonal else
            _etrace_op_name
        )
        self._jitted_call = jax.jit(wrap_etrace_fun(self._call, name))

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
        return f'{self.__class__.__name__}({self.fun.__name__}, is_diagonal={self.is_diagonal})'

    def _call(self, x, weight):
        return self.fun(x, weight)


class StandardOp(ETraceOp):
    """
    The standard operator for the eligibility trace-based online gradient learning.
    """

    def __init__(self, is_diagonal: bool = False):
        super().__init__(self.xw_to_y, is_diagonal=is_diagonal)

    def xw_to_y(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        r"""
        This function is used to compute the output of the operator.

        It computes:

            $$
            y = f(x, w)
            $$

        """
        raise NotImplementedError

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        r"""
        This function is used to compute the weight from the hidden dimensional array.

        It computes:

            $$
            w = f(y, w)
            $$

        This function is mainly used when computing eligibility trace updates based on
        :py:class:`ParamDimVjpAlgorithm`.
        """
        raise NotImplementedError


class MatmulOp(StandardOp):
    """
    The matrix multiplication operator for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

        $$
        y = x @ w + b
        $$

    $b$ is the bias term, which can be optional.
    """

    def __init__(
        self,
        weight_mask: Optional[jax.Array] = None,
    ):
        super().__init__(is_diagonal=False)

        if weight_mask is None:
            pass
        elif isinstance(weight_mask, (np.ndarray, jax.Array, u.Quantity)):
            weight_mask = u.math.asarray(weight_mask)
        else:
            raise TypeError(f'The weight_mask must be an array-like. But got {type(weight_mask)}')
        self.weight_mask = weight_mask

    def xw_to_y(
        self,
        x: bst.typing.ArrayLike,
        w: Dict[str, bst.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

            $$
            y = x @ w + b
            $$

        if the bias is provided.

            $$
            y = x @ w
            $$

        if the bias is not provided.

        """
        if not isinstance(w, dict):
            raise TypeError(f'{MatmulOp.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')
        y = u.math.matmul(x, w['weight'] * self.weight_mask)
        if 'bias' in w:
            y = y + w['bias']
        return y

    def yw_to_w(
        self,
        hidden_dim_arr: bst.typing.ArrayLike,
        weight_dim_tree: Dict[str, bst.typing.ArrayLike],
    ) -> Dict[str, bst.typing.ArrayLike]:
        r"""
        This function is used to compute the weight from the hidden dimensional array.
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


class AbsMatmulOp(MatmulOp):
    """
    The matrix multiplication operator with absolute weight values for eligibility trace-based gradient learning.

    This operator is used to compute the output of the operator, mathematically:

        $$
        y = x @ w + b
        $$

    $b$ is the bias term, which can be optional.

    """

    def __init__(
        self,
        weight_mask: Optional[jax.Array] = None,
    ):
        super().__init__(weight_mask)

    def xw_to_y(
        self,
        x: bst.typing.ArrayLike,
        w: Dict[str, bst.typing.ArrayLike]
    ):
        r"""
        This function is used to compute the output of the operator, mathematically:

            $$
            y = x @ |w| + b
            $$

        if the bias is provided.

            $$
            y = x @ |w|
            $$

        if the bias is not provided.

        """
        if not isinstance(w, dict):
            raise TypeError(f'{MatmulOp.__name__} only supports '
                            f'the dictionary weight. But got {type(w)}')
        if 'weight' not in w:
            raise ValueError(f'The weight must contain the key "weight".')
        y = u.math.matmul(x, u.math.abs(w['weight']) * self.weight_mask)
        if 'bias' in w:
            y = y + w['bias']
        return y


class ElemWiseOp(StandardOp):
    """
    The element-wise operator for the eligibility trace-based gradient learning.
    
    This interface can be used to define any element-wise operation between weight parameters and hidden states. 
    
    Args:
        fn: the element-wise function, which must have the signature: ``(w: ndarray) -> ndarray``.
    """

    def __init__(
        self,
        fn: Callable = lambda w: w,
    ):
        self._raw_fn = fn
        super().__init__(is_diagonal=True)
        self._jitted_call = jax.jit(wrap_etrace_fun(self._call, _etrace_op_name_enable_grad_elem))

    def __call__(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        return self._jitted_call(inputs, weights)

    def xw_to_y(
        self,
        inputs: X,
        weights: W,
    ) -> Y:
        return self._raw_fn(inputs, weights)

    def yw_to_w(
        self,
        hidden_dim_arr: Y,
        weight_dim_tree: W,
    ) -> W:
        prim, f_vjp = jax.vjp(self._raw_fn, weight_dim_tree)
        assert hidden_dim_arr.shape == prim.shape, (
            f'The shape of the hidden_dim_arr must be the same as the weight_dim_tree. '
            f'Got {hidden_dim_arr.shape} and {prim.shape}'
        )
        return f_vjp((hidden_dim_arr,))


class OldStandardETraceOp(ETraceOp):
    """
    The standard operator for the eligibility trace, which is used
    for computing the parameter-dimensional eligibility trace updates.
    """

    def etrace_update(
        self,
        mode: bst.mixin.Mode,
        w: W,
        dh_to_dw: List[W],
        diag_jac: List[jax.Array],
        ph_to_pwx: jax.Array,
        ph_to_pwy: jax.Array
    ):
        r"""
        Standard operator for computing the eligibility trace updates.

        Update: ``eligibility trace`` * ``diagonal hidden Jacobian`` + ``new hidden-to-weight Jacobian``

        .. math::
           d\epsilon^t = D_h ⊙ d\epsilon^{t-1} + df^t

        where :math:`D_h` is the hidden-to-hidden Jacobian diagonal matrix，
        :math:`df^t` is the hidden-to-weight Jacobian matrix.

        For example::

          ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ... + ∂V^t/∂θ1^t


        Args:
            mode: the mode of the operation, which may contain the batching information.
            w: the weight value.
            dh_to_dw: the hidden-to-weight Jacobian.
            diag_jac: the diagonal Jacobian $\frac{ \partial h^t } { \partial h^{t-1} } $ of the hidden states.
            ph_to_pwx: the partial derivative of the hidden with respect to the weight operation input,
                    i.e., the input $x^t$.
            ph_to_pwy: the partial derivative of the hidden with respect to the weight operation output,
                    i.e., the output $\frac{ \partial h^t } { \partial y^t }$.

        """
        raise NotImplementedError

    def hidden_to_etrace(
        self,
        mode: bst.mixin.Mode,
        w: W,
        dl_to_dh: jax.Array,
        dh_to_dw: W
    ):
        r"""
        Compute the gradient of the loss with respect to the weight operation.

        This function is used to merge the hidden dimensional gradients (i.e. the loss-to-hidden
        gradient) into the eligibility trace updates.

        .. math::

           dL/dW = (dL/dH) \circ (dH / dW) \approx \frac{ \partial L^t } { \partial h^t } \circ \epsilon^t

        Args:
            mode: the mode of the operation, which may contain the batching information.
            w: the weight value.
            dl_to_dh: the derivative of the loss with respect to the hidden
                states, i.e., $\frac{ \partial L^t } { \partial h^t }$.
            dh_to_dw: the derivative of the hidden states with respect to the weight operation,
                i.e., the eligibility trace $\frac{ \partial h^t } { \partial W^t } \approx \epsilon^t$.

        """
        raise NotImplementedError


class GeneralETraceOpOld(OldStandardETraceOp):
    """
    The general operator for computing the eligibility trace updates, which can be applied to any :py:class:`ETraceOp`,
    but does not guarantee the computational efficiency.
    """

    def __init__(
        self,
        op: Callable[[X, W], Y],
        xinfo: jax.ShapeDtypeStruct,
        is_diagonal: bool = False
    ):
        super().__init__(op, is_diagonal=is_diagonal)
        #
        # calling the operator through:
        #       y = op(x, w)
        #
        self.op = op
        #
        # the shape and dtype of the input x
        #
        self.xinfo = xinfo

    def etrace_update(
        self,
        mode: bst.mixin.Mode,
        w: W,
        dh_to_dw: List[W],
        diag_jac: List[jax.Array],
        ph_to_pwx: jax.Array,
        ph_to_pwy: Optional[jax.Array],
    ):
        """
        This is the general method for computing the eligibility trace updates, which
        can be applied to any :py:class:`ETraceOp`, but does not guarantee the computational efficiency.

        See the :meth:`StandardETraceOp.etrace_update` for more details.
        """

        assert isinstance(dh_to_dw, (list, tuple)), f'The dh_to_dw must be a list of pytrees. Got {type(dh_to_dw)}'
        assert isinstance(diag_jac, (list, tuple)), f'The diag_jac must be a list of jax.Array. Got {type(diag_jac)}'
        assert len(dh_to_dw) == len(diag_jac), (
            f'The length of dh_to_dw and diag_jac must be the same. '
            f'Got {len(dh_to_dw)} and {len(diag_jac)}'
        )

        #
        # Step 1:
        #
        # update the eligibility trace * hidden diagonal Jacobian
        #         dϵ^t = D_h ⊙ dϵ^t-1, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
        #
        final_dw = None
        for dw, diag in zip(dh_to_dw, diag_jac):
            #
            # convert the diagonal hidden-to-hidden Jacobian to the
            # dimension of weights
            dg_weight = self.dy_to_weight(
                mode,
                w,
                self.op,
                self.xinfo,
                diag
            )

            #
            # compute the element-wise multiplication of:
            #      diagonal * \epsilon (dh_to_dw)
            diag_mul_dw = jax.tree.map(u.math.multiply, dg_weight, dw)
            final_dw = (
                diag_mul_dw
                if final_dw is None else
                jax.tree.map(u.math.add, final_dw, diag_mul_dw)
            )

        #
        # Step 2:
        #
        # update: eligibility trace * hidden diagonal Jacobian + new hidden df
        #        dϵ^t = D_h ⊙ dϵ^t-1 + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
        #
        if ph_to_pwy is not None:
            current_etrace = self.dx_dy_to_weight(
                mode,
                w,
                self.op,
                ph_to_pwx,
                ph_to_pwy
            )
            final_dw = jax.tree.map(u.math.add, final_dw, current_etrace)
        return final_dw

    def hidden_to_etrace(
        self,
        mode: bst.mixin.Mode,
        w: bst.typing.PyTree,
        dl_to_dh: jax.Array,
        dh_to_dw: bst.typing.PyTree
    ):
        """
        This is the general method for computing the gradient of the loss with respect to the weight operation.
        It can be applied to any :py:class:`ETraceOp`, but does not guarantee the computational efficiency.

        See the :meth:`StandardETraceOp.hidden_to_etrace` for more details.
        """

        # compute: dL/dW = (dL/dH) \circ (dH / dW)
        dg_weight = self.dy_to_weight(
            mode,
            w,
            self.op,
            self.xinfo,
            dl_to_dh
        )
        return jax.tree.map(u.math.multiply, dg_weight, dh_to_dw)

    @staticmethod
    def dy_to_weight(
        mode: bst.mixin.Mode,
        weight_vals: bst.typing.PyTree,
        op: Callable,
        x_info: jax.ShapeDtypeStruct,
        dg_hidden: jax.Array
    ) -> bst.typing.PyTree:
        #
        # [KEY]
        # For the following operation:
        #      dL/dW = (dL/dH) \circ (dH / dW)
        #   or
        #      \partial H^t/\partial W = \partial H^t/\partial H^{t-1} \cdot \partial H^{t-1}/\partial W
        #
        # we can compute the gradient of the weight using the following two merging operations:
        #

        # input
        x_data = u.math.ones(x_info.shape, x_info.dtype)

        # transform
        def fn4vjp(dh, x):
            primals, f_vjp = jax.vjp(partial(op, x), weight_vals)
            if isinstance(dh, (tuple, list)):
                assert isinstance(primals, (tuple, list))
                dh = (
                    u.maybe_decimal(u.get_magnitude(dh[0]) * u.get_unit(primals[0])),
                )
            else:
                dh = u.maybe_decimal(u.get_magnitude(dh) * u.get_unit(primals))
            return f_vjp(dh)[0]

        # fun = lambda dh, x: jax.vjp(partial(op, x), weight_path_to_vals)[1](dh)[0]
        if mode.has(bst.mixin.Batching):
            # TODO:
            #    assuming the batch size is the first dimension
            x_data = u.math.expand_dims(x_data, axis=1)
            dg_hidden = u.math.expand_dims(dg_hidden, axis=1)
            dg_weight = jax.vmap(fn4vjp)(dg_hidden, x_data)
        else:
            dg_weight = fn4vjp(dg_hidden, x_data)
        return dg_weight

    @staticmethod
    def dx_dy_to_weight(
        mode: bst.mixin.Mode,
        weight_vals: bst.typing.PyTree,
        op: Callable,
        dg_x: jax.Array,
        dg_y: jax.Array
    ) -> bst.typing.PyTree:
        #
        # [KEY]
        # For the following operation:
        #      dW = dy \otimes dx
        #
        # we can compute the gradient of the weight using the following two merging operations:
        #
        def fn4vjp(dx, dy):
            primals, f_vjp = jax.vjp(partial(op, dx), weight_vals)
            if isinstance(primals, u.Quantity) and isinstance(dy, u.Quantity):
                assert primals.unit.has_same_dim(dy.unit), (
                    f'The unit of the primal and the derivative must '
                    f'be the same. But we got {primals.unit} and {dy.unit}'
                )
            elif isinstance(primals, u.Quantity):
                dy = u.Quantity(dy, unit=primals.unit)
            elif isinstance(dy, u.Quantity):
                raise ValueError(f'The primal must be a quantity. Got {type(primals)}')
            return f_vjp(dy)[0]

        if mode.has(bst.mixin.Batching):
            # TODO:
            #    assuming the batch size is the first dimension
            dg_weight = jax.vmap(fn4vjp)(u.math.expand_dims(dg_x, axis=1),
                                         u.math.expand_dims(dg_y, axis=1))
        else:
            dg_weight = fn4vjp(dg_x, dg_y)
        return dg_weight


class MatMulETraceOpOld(OldStandardETraceOp):
    """
    The standard matrix multiplication operator for the eligibility trace updates.

    This operator is much more efficient than the :py:class:`GeneralETraceOp` for the matrix multiplication operation.

    """

    def __init__(
        self,
        weight_mask: Optional[jax.Array] = None,
        is_diagonal: bool = False
    ):
        super().__init__(self._operation, is_diagonal=is_diagonal)
        self.weight_mask = weight_mask

    def _format_weight(
        self,
        weights,
        keep_unit: bool = True
    ) -> Tuple[Tuple[jax.Array, Optional[jax.Array]], Callable]:
        weights = (weights['weight'], weights.get('bias', None))

        if keep_unit:
            unflatten = lambda weight, bias: (
                {'weight': weight, } if bias is None else
                {'weight': weight, 'bias': bias}
            )
        else:
            w_unit = u.get_unit(weights[0])
            b_unit = u.get_unit(weights[1])

            def unflatten(weight, bias):
                weight = u.maybe_decimal(weight * w_unit)
                bias = None if bias is None else u.maybe_decimal(bias * b_unit)
                if bias is None:
                    return {'weight': weight}
                return {'weight': weight, 'bias': bias}

            weights = tuple([u.get_magnitude(w) for w in weights])
        return weights, unflatten

    def _operation(self, x, w):
        (weight, bias), _ = self._format_weight(w, keep_unit=True)
        if self.weight_mask is not None:
            weight = weight * self.weight_mask
        if bias is None:
            return u.math.matmul(x, weight)
        else:
            return u.math.matmul(x, weight) + bias

    def etrace_update(
        self,
        mode: bst.mixin.Mode,
        w: bst.typing.PyTree,
        dh_to_dw: List[bst.typing.PyTree],
        diag_jac: List[jax.Array],
        ph_to_pwx: jax.Array,
        ph_to_pwy: Optional[jax.Array],
    ):
        """
        This is the standard method for computing the eligibility trace updates for the matrix multiplication operation.

        See the :meth:`StandardETraceOp.etrace_update` for more details.
        """

        # 1. w: the wight value, a pytree
        # 2. dh_to_dw: derivative of hidden to weight, the number equals to the number of hidden states
        # 3. diag_jac: the diagonal Jacobian of the hidden states, the number equals to the number of hidden states
        # 4. ph_to_pwx: the partial derivative of the hidden with respect to the weight input
        # 5. ph_to_pwy: the partial derivative of the hidden with respect to the weight output

        assert isinstance(dh_to_dw, (list, tuple)), f'The dh_to_dw must be a list of pytrees. Got {type(dh_to_dw)}'
        assert isinstance(diag_jac, (list, tuple)), f'The diag_jac must be a list of jax.Array. Got {type(diag_jac)}'
        assert len(dh_to_dw) == len(diag_jac), (
            f'The length of dh_to_dw and diag_jac must be the same. '
            f'Got {len(dh_to_dw)} and {len(diag_jac)}'
        )

        # diag_jac = remove_units(diag_jac)
        # dh_to_dw = remove_units(dh_to_dw)
        ph_to_pwx = remove_units(ph_to_pwx)
        ph_to_pwy = remove_units(ph_to_pwy)

        diag_mul_dhdw = [
            self.hidden_to_etrace(mode, w, dh, dw)
            for dh, dw in zip(diag_jac, dh_to_dw)
        ]
        diag_mul_dhdw = jax.tree.map(lambda *xs: reduce(u.math.add, xs), *diag_mul_dhdw)

        (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(diag_mul_dhdw, keep_unit=False)
        if ph_to_pwy is not None:
            if mode.has(bst.mixin.Batching):
                # dh_to_dweight: (batch_size, input_size, hidden_size,)
                # dh_to_dbias: (batch_size, hidden_size,)
                # ph_to_pwx: (batch_size, input_size,)
                # ph_to_pwy: (batch_size, hidden_size,)
                # dh_to_dweight = dh_to_dweight + u.math.einsum('bi,bh->bih', ph_to_pwx, ph_to_pwy)
                dW = u.math.einsum('bi,bh->bih', ph_to_pwx, ph_to_pwy)
            else:
                # dh_to_dweight: (input_size, hidden_size,)
                # dh_to_dbias: (hidden_size,)
                # ph_to_pwx: (input_size,)
                # ph_to_pwy: (hidden_size,)
                dW = u.math.outer(ph_to_pwx, ph_to_pwy)

            if self.weight_mask is not None:
                dW = dW * self.weight_mask
            dh_to_dweight = dh_to_dweight + dW
            if dh_to_dbias is not None:
                # dh_to_dbias = dh_to_dbias + ph_to_pwy
                dh_to_dbias = dh_to_dbias + ph_to_pwy
        return unflatten(dh_to_dweight, dh_to_dbias)

    def hidden_to_etrace(
        self,
        mode: bst.mixin.Mode,
        w: bst.typing.PyTree,
        dl_to_dh: jax.Array,
        dh_to_dw: bst.typing.PyTree
    ):
        """
        This is the standard method for computing the gradient of the loss with respect to the weight operation
        for the matrix multiplication operation.

        See the :meth:`StandardETraceOp.hidden_to_etrace` for more details.
        """

        # 1. w: the wight value
        # 2. dl_to_dh: the derivative of the loss with respect to the hidden
        # 3. dh_to_dw: the derivative of the hidden with respect to the weight

        (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(dh_to_dw)
        dl_to_dh = remove_units(dl_to_dh)

        if mode.has(bst.mixin.Batching):
            # dl_to_dh: (batch_size, hidden_size,)
            # dh_to_dw: (batch_size, input_size, hidden_size,)
            dh_to_dweight = u.math.expand_dims(dl_to_dh, axis=1) * dh_to_dweight
            if dh_to_dbias is not None:
                dh_to_dbias = dh_to_dbias * dl_to_dh

        else:
            # dl_to_dh: (hidden_size,)
            # dh_to_dw: (input_size, hidden_size,)
            dh_to_dweight = dh_to_dweight * u.math.expand_dims(dl_to_dh, axis=0)
            if dh_to_dbias is not None:
                dh_to_dbias = dh_to_dbias * dl_to_dh

        # weight mask
        if self.weight_mask is not None:
            dh_to_dweight = dh_to_dweight * self.weight_mask
        return unflatten(dh_to_dweight, dh_to_dbias)


class Conv2DETraceOpOld(OldStandardETraceOp):
  """
  The etrace operator for the 2D convolution.

  """

  def __init__(
      self,
      weight_mask: Optional[jax.Array] = None,
      is_diagonal: bool = False
  ):
    super().__init__(fun=self._operation, is_diagonal=is_diagonal)
    self.weight_mask = weight_mask

  def _operation(self, x, w):
    weight, bias = w
    if self.weight_mask is not None:
      weight = weight * self.weight_mask
    return jax.lax.conv_general_dilated(x, weight, (1, 1), 'SAME') + bias

