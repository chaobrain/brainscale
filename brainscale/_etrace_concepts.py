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
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
from typing import Callable, Sequence, Tuple, List, Optional

import jax.lax
import brainstate as bst

from ._misc import BaseEnum
from ._typing import PyTree

__all__ = [
  # eligibility trace related concepts
  'ETraceVar',  # the hidden state for the etrace-based learning
  'ETraceParam',  # the parameter/weight for the etrace-based learning
  'ETraceOp',  # the operator for the etrace-based learning
  'ETraceParamOp',  # the parameter and operator for the etrace-based learning, combining ETraceParam and ETraceOp
  'NoTempParamOp',  # the parameter state with an associated operator
  'NoGradParamOp',
  'stop_param_gradients',
]

_stop_param_gradient = False
_etrace_op_name = '_etrace_weight_operator_call_'
_etrace_op_name_enable_grad = '_etrace_weight_operator_call_enable_grad_'


def wrap_etrace_fun(fun, name: str = _etrace_op_name):
  fun.__name__ = name
  return fun


def is_etrace_op(jit_param_name: str):
  return jit_param_name.startswith(_etrace_op_name)


def is_etrace_op_enable_gradient(jit_param_name: str):
  return jit_param_name.startswith(_etrace_op_name_enable_grad)


# -------------------------------------------------------------------------------------- #
# Eligibility Trace Related Concepts
# -------------------------------------------------------------------------------------- #


class ETraceVar(bst.ShortTermState):
  """
  The Eligibility Trace Hidden Variable.

  Args:
    value: The value of the hidden variable. Currently only support a `jax.Array`.
  """
  __module__ = 'brainscale'

  def __init__(self, value: jax.Array):
    super().__init__(value)
    assert isinstance(self.value, jax.Array), f'Currently, {ETraceVar.__name__} only supports jax.Array.'
    self._check_tree = False


class ETraceParam(bst.ParamState):
  """
  The Eligibility Trace Weight.

  Args:
    value: The value of the weight. Can be a PyTree.
  """
  __module__ = 'brainscale'

  is_not_etrace: bool

  def __init__(self, value: PyTree):
    super().__init__(value)

    self.is_not_etrace = False


class ETraceOp:
  """
  The Eligibility Trace Operator.

  The function must have the signature: ``(x: jax.Array, weight: PyTree) -> jax.Array``.

  Attributes:
    fun: The operator function.
    is_diagonal: bool. Whether the operator is in the hidden diagonal or not.

  Args:
    fun: The operator function.
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
    name = _etrace_op_name_enable_grad if is_diagonal else _etrace_op_name

    def _call(x, weight):
      return self.fun(x, weight)

    self._jitted_call = jax.jit(wrap_etrace_fun(_call, name))

  def __call__(self, x: jax.Array, weight: PyTree) -> jax.Array:
    y = self._jitted_call(x, weight)
    if _stop_param_gradient and not self.is_diagonal:
      y = jax.lax.stop_gradient(self._jitted_call(x, weight))
    return y


class _ETraceGrad(BaseEnum):
  full = 'full'
  approx = 'approx'
  adaptive = 'adaptive'


class ETraceParamOp(ETraceParam):
  """
  The Eligibility Trace Weight and its Associated Operator.

  Args:
    weight: The weight of the ETrace.
    op: The operator for the ETrace. See `ETraceOp`.
  """
  __module__ = 'brainscale'
  op: ETraceOp  # operator

  def __init__(
      self,
      weight: PyTree,
      op: Callable,
      grad: Optional[str] = None,
      is_diagonal: bool = None
  ):
    # weight value
    super().__init__(weight)

    # gradient
    if grad is None:
      grad = 'adaptive'
    assert isinstance(grad, str), f'Currently, {ETraceParamOp.__name__} only supports str.'
    self.gradient = _ETraceGrad.get(grad)

    # operation
    if isinstance(op, ETraceOp):
      self.op = op
      if is_diagonal is not None:
        self.op.is_diagonal = is_diagonal
    else:
      self.op = ETraceOp(op, is_diagonal=is_diagonal if is_diagonal is not None else False)

  def execute(self, x: jax.Array) -> jax.Array:
    return self.op(x, self.value)


class NoTempParamOp(bst.ParamState):
  """
  The Parameter State with an Associated Operator with no temporal dependent back-propagation.

  This class behaves the same as :py:class:`ETraceParamOp`, but will not build the
  eligibility trace graph when using online learning. Therefore, in a sequence
  learning task, the weight can only be trained with the spatial gradients.

  Args:
    value: The value of the parameter.
    op: The operator for the parameter. See `ETraceOp`.
  """
  __module__ = 'brainscale'
  op: Callable  # operator

  def __init__(
      self,
      value: PyTree,
      op: Callable,
  ):
    super().__init__(value)

    # operation
    if isinstance(op, ETraceOp):
      op = op.fun
    self.op = op

  def execute(self, x: jax.Array) -> jax.Array:
    return self.op(x, self.value)


class NoGradParamOp(object):
  """
  The Parameter State with an Associated Operator that does not require to compute gradients.

  Args:
    value: The value of the parameter.
    op: The operator for the parameter.
  """
  __module__ = 'brainscale'

  def __init__(
      self,
      value: PyTree,
      op: Callable
  ):
    super().__init__()

    self.value = value
    if isinstance(op, ETraceOp):
      op = op.fun
      # raise TypeError(f'{NoGradParamOp.__name__} does not support {ETraceOp.__name__}. Please use ETraceOp.fun instead.')
    self.op = op

  def execute(self, x: jax.Array) -> jax.Array:
    return self.op(x, self.value)


# -------------------------------------------------------------------------------------- #


@contextlib.contextmanager
def stop_param_gradients():
  """
  Stop the weight gradients for the ETrace weight operator.

  Example:
    ```
    with stop_weight_gradients():
      # do something
    ```

  """
  global _stop_param_gradient
  try:
    _stop_param_gradient = True
    yield
  finally:
    _stop_param_gradient = False


def assign_state_values(states, state_values):
  """
  Assign values to the states.

  Args:
    states: The states to be assigned.
    state_values: The values to be assigned.
  """
  for st, val in zip(states, state_values):
    st.value = val


def split_states(states: Sequence[bst.State]) -> Tuple[List[bst.ParamState], List[ETraceVar], List[bst.State]]:
  """
  Split the states into weight states, hidden states, and other states.

  Args:
    states: The states to be split.

  Returns:
    param_states: The weight parameter states.
    hidden_states: The hidden states.
    other_states: The other states.

  """
  param_states, hidden_states, other_states = [], [], []
  for st in states:
    if isinstance(st, ETraceVar):  # etrace hidden variables
      hidden_states.append(st)
    elif isinstance(st, bst.ParamState):  # including all weight states, ParamState, ETraceParam
      param_states.append(st)
    else:
      other_states.append(st)
  return param_states, hidden_states, other_states


def split_states_v2(
    states: Sequence[bst.State]
) -> Tuple[List[ETraceParam], List[ETraceVar], List[bst.ParamState], List[bst.State]]:
  """
  Split the states into weight states, hidden states, and other states.

  .. note::

      This function is important since it determines what ParamState should be
      trained with the eligibility trace and what should not.

  Args:
    states: The states to be split.

  Returns:
    etrace_param_states: The etrace parameter states.
    hidden_states: The hidden states.
    param_states: The other kinds of parameter states.
    other_states: The other states.
  """
  etrace_param_states, hidden_states, param_states, other_states = [], [], [], []
  for st in states:
    if isinstance(st, ETraceVar):
      hidden_states.append(st)
    elif isinstance(st, ETraceParam):
      if st.is_not_etrace:
        # The ETraceParam is set to "is_not_etrace" since
        # no hidden state is associated with it,
        # so it should be treated as a normal parameter state
        # and be trained with spatial gradients only
        param_states.append(st)
      else:
        etrace_param_states.append(st)
    else:
      if isinstance(st, bst.ParamState):
        # The ParamState which is not an ETraceParam,
        # should be treated as a normal parameter state
        # and be trained with spatial gradients only
        param_states.append(st)
      else:
        other_states.append(st)
  return etrace_param_states, hidden_states, param_states, other_states
