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

from functools import partial, reduce
from typing import Tuple, Callable, List, Optional, Any

import brainstate as bst
import brainunit as u
import jax
import jax.core

from ._etrace_concepts import ETraceOp
from ._typing import PyTree

WeightTree = Any

__all__ = [
  'StandardETraceOp',
  'GeneralETraceOp',
  'MatMulETraceOp',
]


class StandardETraceOp(ETraceOp):
  """
  The standard operator for the eligibility trace, which is used
  for computing the parameter-dimensional eligibility trace updates.
  """

  def etrace_update(
      self,
      mode: bst.mixin.Mode,
      w: WeightTree,
      dh_to_dw: List[WeightTree],
      diag_jac: List[jax.Array],
      ph_to_pwx: jax.Array,
      ph_to_pwy: jax.Array
  ):
    r"""
    Compute the eligibility trace updates.

    Update: ``eligibility trace`` * ``hidden diagonal Jacobian`` + ``new hidden-weight Jacobian``

    .. math::
       d\epsilon^t = D_h ⊙ d\epsilon^{t-1} + df^t

    where :math:`D_h` is the hidden-to-hidden Jacobian diagonal matrix，
    :math:`df^t` is the hidden-to-weight Jacobian matrix.

    For example::

      ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ... + ∂V^t/∂θ1^t

    """
    raise NotImplementedError

  def hidden_to_etrace(
      self,
      mode: bst.mixin.Mode,
      w: WeightTree,
      dl_to_dh: jax.Array,
      dh_to_dw: WeightTree
  ):
    r"""
    Compute the hidden-to-etrace updates.

    This function is used to merge the hidden dimensional gradients into the
    parameter-dimensional gradients. For example:

    .. math::

       dL/dW = (dL/dH) \circ (dH / dW)

    """
    raise NotImplementedError


class GeneralETraceOp(StandardETraceOp):
  """
  The general operator for computing the eligibility trace updates.

  This operator can be applied to any operation, but does not guarantee the
  computational efficiency.
  """

  def __init__(
      self,
      op: Callable,
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
      w: WeightTree,
      dh_to_dw: List[WeightTree],
      diag_jac: List[jax.Array],
      ph_to_pwx: jax.Array,
      ph_to_pwy: Optional[jax.Array],
  ) -> WeightTree:
    """
    See the :meth:`StandardETraceOp.etrace_update` for more details.
    """

    assert isinstance(dh_to_dw, (list, tuple)), f'The dh_to_dw must be a list of pytrees. Got {type(dh_to_dw)}'
    assert isinstance(diag_jac, (list, tuple)), f'The diag_jac must be a list of jax.Array. Got {type(diag_jac)}'
    assert len(dh_to_dw) == len(diag_jac), (f'The length of dh_to_dw and diag_jac must be the same. '
                                            f'Got {len(dh_to_dw)} and {len(diag_jac)}')
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
      dg_weight = self.dy_to_weight(mode,
                                    w,
                                    self.op,
                                    self.xinfo,
                                    diag)
      #
      # compute the element-wise multiplication of:
      #      diagonal * \epsilon (dh_to_dw)
      diag_mul_dw = jax.tree.map(u.math.multiply, dg_weight, dw)
      final_dw = diag_mul_dw if final_dw is None else jax.tree.map(u.math.add, final_dw, diag_mul_dw)

    #
    # Step 2:
    #
    # update: eligibility trace * hidden diagonal Jacobian + new hidden df
    #        dϵ^t = D_h ⊙ dϵ^t-1 + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
    #
    if ph_to_pwy is not None:
      current_etrace = self.dx_dy_to_weight(mode,
                                            w,
                                            self.op,
                                            ph_to_pwx,
                                            ph_to_pwy)
      final_dw = jax.tree.map(u.math.add, final_dw, current_etrace)
    return final_dw

  def hidden_to_etrace(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dl_to_dh: jax.Array,
      dh_to_dw: PyTree
  ):
    """
    See the :meth:`StandardETraceOp.hidden_to_etrace` for more details.
    """

    # compute: dL/dW = (dL/dH) \circ (dH / dW)
    dg_weight = self.dy_to_weight(mode,
                                  w,
                                  self.op,
                                  self.xinfo,
                                  dl_to_dh)
    return jax.tree.map(u.math.multiply, dg_weight, dh_to_dw)

  @staticmethod
  def dy_to_weight(
      mode: bst.mixin.Mode,
      weight_vals: PyTree,
      op: Callable,
      x_info: jax.ShapeDtypeStruct,
      dg_hidden: jax.Array
  ) -> PyTree:
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
    fun = lambda dh, x: jax.vjp(partial(op, x), weight_vals)[1](dh)[0]
    if mode.has(bst.mixin.Batching):
      # TODO:
      #    assuming the batch size is the first dimension
      x_data = u.math.expand_dims(x_data, axis=1)
      dg_hidden = u.math.expand_dims(dg_hidden, axis=1)
      dg_weight = jax.vmap(fun)(dg_hidden, x_data)
    else:
      dg_weight = fun(dg_hidden, x_data)
    return dg_weight

  @staticmethod
  def dx_dy_to_weight(
      mode: bst.mixin.Mode,
      weight_vals: PyTree,
      op: Callable,
      dg_x: jax.Array,
      dg_y: jax.Array
  ) -> PyTree:
    #
    # [KEY]
    # For the following operation:
    #      dW = dy \otimes dx
    #
    # we can compute the gradient of the weight using the following two merging operations:
    #
    fun = lambda dx, dy: jax.vjp(partial(op, dx), weight_vals)[1](dy)[0]
    if mode.has(bst.mixin.Batching):
      # TODO:
      #    assuming the batch size is the first dimension
      dg_weight = jax.vmap(fun)(u.math.expand_dims(dg_x, axis=1),
                                u.math.expand_dims(dg_y, axis=1))
    else:
      dg_weight = fun(dg_x, dg_y)
    return dg_weight


def binary_op(op, x, y):
  if isinstance(x, u.Quantity) and isinstance(y, u.Quantity):
    return op(x, y)
  if isinstance(x, u.Quantity):
    return u.Quantity(op(x.magnitude, y), unit=x.unit)
  if isinstance(y, u.Quantity):
    return u.Quantity(op(x, y.magnitude), unit=y.unit)
  return op(x, y)


class MatMulETraceOp(StandardETraceOp):
  """
  The standard matrix multiplication operator for the eligibility trace.

  """

  def __init__(
      self,
      weight_mask: Optional[jax.Array] = None,
      is_diagonal: bool = False
  ):
    super().__init__(self._operation, is_diagonal=is_diagonal)
    self.weight_mask = weight_mask

  def _format_weight(self, weights) -> Tuple[Tuple[jax.Array, Optional[jax.Array]], Callable]:
    if isinstance(weights, dict):
      weights = (weights['weight'], weights.get('bias', None))
      unflatten = lambda w, b: {'weight': (w), 'bias': b} if (b is not None) else {'weight': w}
    elif isinstance(weights, (tuple, list)):
      weights = (weights[0], weights[1] if len(weights) > 1 else None)
      unflatten = lambda w, b: (w, b) if (b is not None) else (w,)
    elif isinstance(weights, jax.Array):
      weights = (weights, None)
      unflatten = lambda w, b: w if (b is None) else (w, b)
    else:
      raise ValueError(f'Invalid weight type: {type(weights)}')
    # weights = jax.tree.map(_get_mantissa, weights, is_leaf=_is_quantity)
    # units = jax.tree.map(_get_unit, weights, is_leaf=_is_quantity)
    return weights, unflatten

  def _operation(self, x, w):
    (weight, bias), _ = self._format_weight(w)
    if self.weight_mask is not None:
      weight = weight * self.weight_mask
    if bias is None:
      return u.math.matmul(x, weight)
    else:
      return u.math.matmul(x, weight) + bias

  def etrace_update(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dh_to_dw: List[PyTree],
      diag_jac: List[jax.Array],
      ph_to_pwx: jax.Array,
      ph_to_pwy: Optional[jax.Array],
  ):
    """
    See the :meth:`StandardETraceOp.etrace_update` for more details.
    """

    # 1. w: the wight value, a pytree
    # 2. dh_to_dw: derivative of hidden to weight, the number equals to the number of hidden states
    # 3. diag_jac: the diagonal Jacobian of the hidden states, the number equals to the number of hidden states
    # 4. ph_to_pwx: the partial derivative of the hidden with respect to the weight input
    # 5. ph_to_pwy: the partial derivative of the hidden with respect to the weight output

    assert isinstance(dh_to_dw, (list, tuple)), f'The dh_to_dw must be a list of pytrees. Got {type(dh_to_dw)}'
    assert isinstance(diag_jac, (list, tuple)), f'The diag_jac must be a list of jax.Array. Got {type(diag_jac)}'
    assert len(dh_to_dw) == len(diag_jac), (f'The length of dh_to_dw and diag_jac must be the same. '
                                            f'Got {len(dh_to_dw)} and {len(diag_jac)}')

    diag_mul_dhdw = [self.hidden_to_etrace(mode, w, dh, dw)
                     for dh, dw in zip(diag_jac, dh_to_dw)]
    diag_mul_dhdw = jax.tree.map(lambda *xs: reduce(u.math.add, xs), *diag_mul_dhdw)

    (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(diag_mul_dhdw)
    if ph_to_pwy is not None:
      if mode.has(bst.mixin.Batching):
        # dh_to_dweight: (batch_size, input_size, hidden_size,)
        # dh_to_dbias: (batch_size, hidden_size,)
        # ph_to_pwx: (batch_size, input_size,)
        # ph_to_pwy: (batch_size, hidden_size,)
        # dh_to_dweight = dh_to_dweight + u.math.einsum('bi,bh->bih', ph_to_pwx, ph_to_pwy)
        dh_to_dweight = binary_op(jax.numpy.add, dh_to_dweight, u.math.einsum('bi,bh->bih', ph_to_pwx, ph_to_pwy))
        if dh_to_dbias is not None:
          # dh_to_dbias = dh_to_dbias + ph_to_pwy
          dh_to_dbias = binary_op(jax.numpy.add, dh_to_dbias, ph_to_pwy)
      else:
        # dh_to_dweight: (input_size, hidden_size,)
        # dh_to_dbias: (hidden_size,)
        # ph_to_pwx: (input_size,)
        # ph_to_pwy: (hidden_size,)
        dh_to_dweight = binary_op(jax.numpy.add, dh_to_dweight, u.math.outer(ph_to_pwx, ph_to_pwy))
        if dh_to_dbias is not None:
          dh_to_dbias = binary_op(jax.numpy.add, dh_to_dbias, ph_to_pwy)
    return unflatten(dh_to_dweight, dh_to_dbias)

  def hidden_to_etrace(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dl_to_dh: jax.Array,
      dh_to_dw: PyTree
  ):
    """
    See the :meth:`StandardETraceOp.hidden_to_etrace` for more details.
    """

    # 1. w: the wight value
    # 2. dl_to_dh: the derivative of the loss with respect to the hidden
    # 3. dh_to_dw: the derivative of the hidden with respect to the weight

    (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(dh_to_dw)
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
    return unflatten(dh_to_dweight, dh_to_dbias)

# class AbsMatMulETraceOp(MatMulETraceOp):
#   """
#   The standard matrix multiplication operator for the eligibility trace.
#
#   """
#
#   def __init__(
#       self,
#       weight_mask: Optional[jax.Array] = None,
#       is_diagonal: bool = False
#   ):
#     super().__init__(weight_mask, is_diagonal=is_diagonal)
#
#   def _operation(self, x, w):
#     (weight, bias), _ = self._format_weight(w)
#     weight = jnp.abs(weight)
#     if self.weight_mask is not None:
#       weight = weight * self.weight_mask
#     if bias is None:
#       return jnp.matmul(x, weight)
#     else:
#       return jnp.matmul(x, weight) + bias
#
#
# class Conv2dETraceOp(StandardETraceOp):
#   """
#   The etrace operator for the 2D convolution.
#
#   """
#
#   def __init__(
#       self,
#       weight_mask: Optional[jax.Array] = None,
#       is_diagonal: bool = False
#   ):
#     super().__init__(fun=self._operation, is_diagonal=is_diagonal)
#     self.weight_mask = weight_mask
#
#   def _operation(self, x, w):
#     weight, bias = w
#     if self.weight_mask is not None:
#       weight = weight * self.weight_mask
#     return jax.lax.conv_general_dilated(x, weight, (1, 1), 'SAME') + bias
