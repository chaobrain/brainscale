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
from typing import Tuple, Callable, List, Optional

import brainstate as bst
import jax
import jax.core
import jax.numpy as jnp

from ._etrace_concepts import ETraceOp
from .typing import PyTree

__all__ = [
  'StandardETraceOp',
  'GeneralETraceOp',
  'MatMulETraceOp',
  'AbsMatMulETraceOp',
]


class StandardETraceOp(ETraceOp):
  """
  The standard operator for the eligibility trace.
  """

  def etrace_update(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dh_to_dw: List[PyTree],
      diag_jac: List[jax.Array],
      ph_to_pwx: jax.Array,
      ph_to_pwy: jax.Array
  ):
    raise NotImplementedError

  def hidden_to_etrace(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dl_to_dh: jax.Array,
      dh_to_dw: PyTree
  ):
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
      w: PyTree,
      dh_to_dw: List[PyTree],
      diag_jac: List[jax.Array],
      ph_to_pwx: jax.Array,
      ph_to_pwy: jax.Array
  ):

    # compute: diagonal * dh_to_dw
    final_dw = None
    for dw, diag in zip(dh_to_dw, diag_jac):
      dg_weight = self._dy_to_weight(mode,
                                     w,
                                     self.op,
                                     self.xinfo,
                                     diag)
      diag_mul_dw = jax.tree.map(jnp.multiply, dg_weight, dw)
      if final_dw is None:
        final_dw = diag_mul_dw
      else:
        final_dw = jax.tree.map(jnp.add, final_dw, diag_mul_dw)

    # compute: current_etrace
    current_etrace = self._dx_dy_to_weight(mode,
                                           w,
                                           self.op,
                                           ph_to_pwx,
                                           ph_to_pwy)

    # compute: diagonal * dh_to_dw + current_etrace
    new_bwg = jax.tree.map(jnp.add, final_dw, current_etrace)
    return new_bwg

  def hidden_to_etrace(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dl_to_dh: jax.Array,
      dh_to_dw: PyTree
  ):
    # compute: dL/dW = (dL/dH) \circ (dH / dW)
    dg_weight = self._dy_to_weight(mode,
                                   w,
                                   self.op,
                                   self.xinfo,
                                   dl_to_dh)
    return jax.tree.map(jnp.multiply, dg_weight, dh_to_dw)

  @staticmethod
  def _dy_to_weight(
      mode: bst.mixin.Mode,
      weight_vals: PyTree,
      op: Callable,
      xinfo: jax.ShapeDtypeStruct,
      dg_hidden: jax.Array
  ) -> PyTree:
    # [KEY]
    # For the following operation:
    #      dL/dW = (dL/dH) \circ (dH / dW)
    #   or
    #      \partial H/\partial W = \partial H/\partial H \cdot \partial H/\partial W
    #
    # we can compute the gradient of the weight using the following two merging operations:

    # input
    x_data = jnp.ones(xinfo.shape, xinfo.dtype)

    # transform
    fun = lambda dh, x: jax.vjp(partial(op, x), weight_vals)[1](dh)[0]
    if mode.has(bst.mixin.Batching):
      x_data = jnp.expand_dims(x_data, axis=1)
      dg_hidden = jnp.expand_dims(dg_hidden, axis=1)
      dG_hidden_like_weight = jax.vmap(fun)(dg_hidden, x_data)
    else:
      dG_hidden_like_weight = fun(dg_hidden, x_data)

    return dG_hidden_like_weight

  @staticmethod
  def _dx_dy_to_weight(
      mode: bst.mixin.Mode,
      weight_vals: PyTree,
      op: Callable,
      dg_x: jax.Array,
      dg_y: jax.Array
  ) -> PyTree:
    # [KEY]
    # For the following operation:
    #      dW = dy \otimes dx
    #
    # we can compute the gradient of the weight using the following two merging operations:

    fun = lambda dx, dy: jax.vjp(partial(op, dx), weight_vals)[1](dy)[0]
    if mode.has(bst.mixin.Batching):
      dG_weight = jax.vmap(fun)(jnp.expand_dims(dg_x, axis=1), jnp.expand_dims(dg_y, axis=1))
    else:
      dG_weight = fun(dg_x, dg_y)
    return dG_weight


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

  def _format_weight(self, weight) -> Tuple[Tuple[jax.Array, Optional[jax.Array]], Callable]:
    if isinstance(weight, dict):
      weight = (weight['weight'], weight.get('bias', None))
      unflatten = lambda w, b: {'weight': w, 'bias': b} if (b is not None) else {'weight': w}
    elif isinstance(weight, (tuple, list)):
      weight = (weight[0], weight[1] if len(weight) > 1 else None)
      unflatten = lambda w, b: (w, b) if (b is not None) else (w,)
    elif isinstance(weight, jax.Array):
      weight = (weight, None)
      unflatten = lambda w, b: w if (b is None) else (w, b)
    else:
      raise ValueError(f'Invalid weight type: {type(weight)}')
    return weight, unflatten

  def _operation(self, x, w):
    (weight, bias), _ = self._format_weight(w)
    if self.weight_mask is not None:
      weight = weight * self.weight_mask
    if bias is None:
      return jnp.matmul(x, weight)
    else:
      return jnp.matmul(x, weight) + bias

  def etrace_update(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dh_to_dw: List[PyTree],
      diag_jac: List[jax.Array],
      ph_to_pwx: jax.Array,
      ph_to_pwy: jax.Array
  ):

    # 1. w: the wight value, a pytree
    # 2. dh_to_dw: derivative of hidden to weight, the number equals to the number of hidden states
    # 3. diag_jac: the diagonal Jacobian of the hidden states, the number equals to the number of hidden states
    # 4. ph_to_pwx: the partial derivative of the hidden with respect to the weight input
    # 5. ph_to_pwy: the partial derivative of the hidden with respect to the weight output

    diag_mul_dhdw = [self.hidden_to_etrace(mode, w, dh, dw)
                     for dh, dw in zip(diag_jac, dh_to_dw)]
    diag_mul_dhdw = jax.tree.map(lambda *xs: reduce(jnp.add, xs), *diag_mul_dhdw)

    (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(diag_mul_dhdw)
    if mode.has(bst.mixin.Batching):
      # dh_to_dweight: (batch_size, input_size, hidden_size,)
      # dh_to_dbias: (batch_size, hidden_size,)
      # ph_to_pwx: (batch_size, input_size,)
      # ph_to_pwy: (batch_size, hidden_size,)
      dh_to_dweight = dh_to_dweight + jnp.einsum('bi,bh->bih', ph_to_pwx, ph_to_pwy)
      if dh_to_dbias is not None:
        dh_to_dbias = dh_to_dbias + ph_to_pwy
    else:
      # dh_to_dweight: (input_size, hidden_size,)
      # dh_to_dbias: (hidden_size,)
      # ph_to_pwx: (input_size,)
      # ph_to_pwy: (hidden_size,)
      dh_to_dweight = (dh_to_dweight + jnp.outer(ph_to_pwx, ph_to_pwy))
      if dh_to_dbias is not None:
        dh_to_dbias = dh_to_dbias + ph_to_pwy
    return unflatten(dh_to_dweight, dh_to_dbias)

  def hidden_to_etrace(
      self,
      mode: bst.mixin.Mode,
      w: PyTree,
      dl_to_dh: jax.Array,
      dh_to_dw: PyTree
  ):
    # 1. w: the wight value
    # 2. dl_to_dh: the derivative of the loss with respect to the hidden
    # 3. dh_to_dw: the derivative of the hidden with respect to the weight

    (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(dh_to_dw)
    if mode.has(bst.mixin.Batching):
      # dl_to_dh: (batch_size, hidden_size,)
      # dh_to_dw: (batch_size, input_size, hidden_size,)
      dh_to_dweight = jnp.expand_dims(dl_to_dh, axis=1) * dh_to_dweight
      if dh_to_dbias is not None:
        dh_to_dbias = dh_to_dbias * dl_to_dh
    else:
      # dl_to_dh: (hidden_size,)
      # dh_to_dw: (input_size, hidden_size,)
      dh_to_dweight = dh_to_dweight * jnp.expand_dims(dl_to_dh, axis=0)
      if dh_to_dbias is not None:
        dh_to_dbias = dh_to_dbias * dl_to_dh
    return unflatten(dh_to_dweight, dh_to_dbias)


class AbsMatMulETraceOp(MatMulETraceOp):
  """
  The standard matrix multiplication operator for the eligibility trace.

  """

  def __init__(
      self,
      weight_mask: Optional[jax.Array] = None,
      is_diagonal: bool = False
  ):
    super().__init__(weight_mask, is_diagonal=is_diagonal)

  def _operation(self, x, w):
    (weight, bias), _ = self._format_weight(w)
    weight = jnp.abs(weight)
    if self.weight_mask is not None:
      weight = weight * self.weight_mask
    if bias is None:
      return jnp.matmul(x, weight)
    else:
      return jnp.matmul(x, weight) + bias


class Conv2dETraceOp(StandardETraceOp):
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
