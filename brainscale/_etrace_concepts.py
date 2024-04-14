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
import contextlib
import functools
from typing import Callable, Sequence, Tuple, List, Optional

import braincore as bc
import jax.lax

from .typing import PyTree

__all__ = [
  # eligibility trace related concepts
  'ETraceVar',  # the hidden state for RTRL
  'ETraceParam',  # the parameter for RTRL
  'ETraceOp',  # the operator for ETrace
  'ETraceParamOp',  # the parameter with an associated operator for ETrace, combining ETraceParam and ETraceOp
  'NormalParamOp',  # the parameter state with an associated operator
  'stop_param_gradients',
]

_stop_param_gradient = False
_etrace_op_name = '_etrace_weight_operator_call_'


def wrap_etrace_fun(fun):
  fun.__name__ = _etrace_op_name
  return fun


# -------------------------------------------------------------------------------------- #
# Eligibility Trace Related Concepts
# -------------------------------------------------------------------------------------- #


class ETraceVar(bc.State):
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


class ETraceParam(bc.ParamState):
  """
  The Eligibility Trace Weight.

  Args:
    value: The value of the weight. Can be a PyTree.
  """
  __module__ = 'brainscale'


class ETraceOp(object):
  """
  The Eligibility Trace Operator.

  The function must have the signature: ``(x: jax.Array, weight: PyTree) -> jax.Array``.

  Args:
    fun: The operator function.
  """
  __module__ = 'brainscale'

  def __init__(self, fun: Callable):
    self._fun = fun
    self.stop_behavior: Optional[Callable] = None

  @functools.partial(jax.jit, static_argnums=0)
  @wrap_etrace_fun
  def _call(self, x, weight):
    return self._fun(x, weight)

  def __call__(self, x: jax.Array, weight: PyTree) -> jax.Array:
    if _stop_param_gradient:
      if self.stop_behavior is None:
        y = self._call(x, weight)
        y = jax.lax.stop_gradient(y)
      else:
        y = self.stop_behavior(x, weight)
    else:
      y = self._call(x, weight)
    return y


class ETraceParamOp(ETraceParam):
  """
  The Eligibility Trace Weight and its Associated Operator.

  Args:
    weight: The weight of the ETrace.
    op: The operator for the ETrace. See `ETraceOp`.
  """
  __module__ = 'brainscale'
  op: ETraceOp  # operator

  def __init__(self, weight: PyTree, op: Callable):
    super().__init__(weight)

    # operation
    if isinstance(op, ETraceOp):
      self.op = op
    else:
      self.op = ETraceOp(op)

  def execute(self, x: jax.Array) -> jax.Array:
    return self.op(x, self.value)


class NormalParamOp(bc.ParamState):
  """
  The Parameter State with an Associated Operator.

  Args:
    value: The value of the parameter.
    op: The operator for the parameter. See `ETraceOp`.
  """
  __module__ = 'brainscale'
  op: Callable  # operator

  def __init__(self, value: PyTree, op: Callable):
    super().__init__(value)

    # operation
    assert not isinstance(op, ETraceOp), f'{NormalParamOp.__name__} does not support {ETraceOp.__name__}.'
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


def split_states(states: Sequence[bc.State]) -> Tuple[List[bc.ParamState], List[ETraceVar], List[bc.State]]:
  """
  Split the states into weight states, hidden states, and other states.

  Args:
    states: The states to be split.

  Returns:
    weight_states: The weight parameter states.
    hidden_states: The hidden states.
    other_states: The other states.

  """
  param_states, hidden_states, other_states = [], [], []
  for st in states:
    if isinstance(st, ETraceVar):  # etrace hidden variables
      hidden_states.append(st)
    elif isinstance(st, bc.ParamState):  # including all weight states, ParamState, ETraceParam
      param_states.append(st)
    else:
      other_states.append(st)
  return param_states, hidden_states, other_states


def split_states_v2(
    states: Sequence[bc.State]
) -> Tuple[List[ETraceParam], List[ETraceVar], List[bc.ParamState], List[bc.State]]:
  """
  Split the states into weight states, hidden states, and other states.

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
      etrace_param_states.append(st)
    else:
      if isinstance(st, bc.ParamState):
        param_states.append(st)
      else:
        other_states.append(st)
  return etrace_param_states, hidden_states, param_states, other_states

