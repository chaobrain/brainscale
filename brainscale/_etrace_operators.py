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

from typing import Optional, Tuple, Callable

import braincore as bc
import jax
import jax.numpy as jnp

from ._etrace_concepts import ETraceOp

__all__ = [
  'StandardETraceOp',
  'MatMulETraceOp',
]


class StandardETraceOp(ETraceOp):
  """
  The standard operator for the eligibility trace.
  """

  def etrace_update(self, w, dh_to_dw, diag, ph_to_pwx, ph_to_pwy):
    raise NotImplementedError

  def hidden_to_etrace(self, w, dl_to_dh, dh_to_dw):
    raise NotImplementedError


class MatMulETraceOp(StandardETraceOp):
  """
  The standard matrix multiplication operator for the eligibility trace.

  """

  def __init__(self, weight_mask: Optional[jax.Array] = None):
    super().__init__(self._operation)
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

  def etrace_update(self, w, dh_to_dw, diag, ph_to_pwx, ph_to_pwy):
    # 1. w: the wight value
    # 2. dh_to_dw: the derivative of the hidden with respect to the weight
    # 3. diag: the diagonal of the hidden Jacobian
    # 4. ph_to_pwx: the partial derivative of the hidden with respect to the weight input
    # 5. ph_to_pwy: the partial derivative of the hidden with respect to the weight output

    (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(dh_to_dw)
    if self.mode.has(bc.mixin.Batching):
      # dh_to_dw: (batch_size, input_size, hidden_size,)
      # diag: (batch_size, hidden_size,)
      # ph_to_pwx: (batch_size, input_size,)
      # ph_to_pwy: (batch_size, hidden_size,)
      # dh_to_dbias: (batch_size, hidden_size,)
      dh_to_dweight = dh_to_dweight * jnp.expand_dims(diag, axis=1) + jnp.einsum('bi,bh->bih', ph_to_pwx, ph_to_pwy)
      if dh_to_dbias is not None:
        dh_to_dbias = dh_to_dbias * diag + ph_to_pwy
    else:
      # dh_to_dw: (input_size, hidden_size,)
      # diag: (hidden_size,)
      # ph_to_pwx: (input_size,)
      # ph_to_pwy: (hidden_size,)
      # dh_to_dbias: (hidden_size,)
      dh_to_dweight = dh_to_dweight * jnp.expand_dims(diag, axis=0) + jnp.outer(ph_to_pwx, ph_to_pwy)
      if dh_to_dbias is not None:
        dh_to_dbias = dh_to_dbias * diag + ph_to_pwy
    return unflatten(dh_to_dweight, dh_to_dbias)

  def hidden_to_etrace(self, w, dl_to_dh, dh_to_dw):
    # 1. w: the wight value
    # 2. dl_to_dh: the derivative of the loss with respect to the hidden
    # 3. dh_to_dw: the derivative of the hidden with respect to the weight

    (dh_to_dweight, dh_to_dbias), unflatten = self._format_weight(dh_to_dw)
    if self.mode.has(bc.mixin.Batching):
      # dl_to_dh: (batch_size, hidden_size,)
      # dh_to_dw: (batch_size, input_size, hidden_size,)
      # return: (batch_size, input_size, hidden_size,)
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


class Conv2dETraceOp(StandardETraceOp):
  """
  The etrace operator for the 2D convolution.

  """

  def __init__(self, weight_mask: Optional[jax.Array] = None):
    super().__init__(fun=self._operation)
    self.weight_mask = weight_mask

  def _operation(self, x, w):
    weight, bias = w
    if self.weight_mask is not None:
      weight = weight * self.weight_mask
    return jax.lax.conv_general_dilated(x, weight, (1, 1), 'SAME') + bias
