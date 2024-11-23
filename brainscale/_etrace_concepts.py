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
import threading
from typing import Callable, Sequence, Tuple, List, Optional, Hashable, Dict

import brainstate as bst
import brainunit as u
import jax.lax

from ._misc import BaseEnum
from ._typing import (PyTree,
                      Path,
                      WeightVals,
                      HiddenVals,
                      StateVals)

__all__ = [
    # eligibility trace related concepts
    'ETraceState',  # the hidden state for the etrace-based learning
    'ETraceVar',
    'ETraceParam',  # the parameter/weight for the etrace-based learning
    'ETraceOp',  # the operator for the etrace-based learning
    'ETraceParamOp',  # the parameter and operator for the etrace-based learning, combining ETraceParam and ETraceOp
    'NonTempParamOp',  # the parameter state with an associated operator
    'NoGradParamOp',
    'stop_param_gradients',
]

_etrace_op_name = '_etrace_weight_operator_call_'
_etrace_op_name_enable_grad = '_etrace_weight_operator_call_enable_grad_'


class CONTEXT(threading.local):
    def __init__(self):
        self.stop_param_gradient = [False]


context = CONTEXT()


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


class ETraceState(bst.ShortTermState):
    """
    The Eligibility Trace Hidden State.

    Args:
      value: The value of the hidden state. Currently only support a `jax.Array` or `brainunit.Quantity`.
    """
    __module__ = 'brainscale'

    def __init__(self, value: bst.typing.ArrayLike, name: Optional[str] = None):
        super().__init__(value, name=name)
        if not isinstance(self.value, (jax.Array, u.Quantity)):
            raise TypeError(f'Currently, {ETraceState.__name__} only supports jax.Array and brainunit.Quantity. '
                            f'But we got {type(self.value)}.')
        self._check_tree = False


ETraceVar = ETraceState


class ETraceParam(bst.ParamState):
    """
    The Eligibility Trace Weight Parameter.

    Args:
      value: The value of the weight. Can be a PyTree.
      name: The name of the weight.
    """
    __module__ = 'brainscale'

    is_not_etrace: bool

    def __init__(
        self,
        value: PyTree,
        name: Optional[str] = None
    ):
        super().__init__(value, name=name)

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
        name = _etrace_op_name_enable_grad if is_diagonal else _etrace_op_name

        def _call(x, weight):
            return self.fun(x, weight)

        self._jitted_call = jax.jit(wrap_etrace_fun(_call, name))

    def __call__(self, x: jax.Array, weight: PyTree) -> jax.Array:
        y = self._jitted_call(x, weight)
        if context.stop_param_gradient[-1] and not self.is_diagonal:
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
        is_diagonal: bool = None,
        name: Optional[str] = None
    ):
        # weight value
        super().__init__(weight, name=name)

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


class NonTempParamOp(bst.ParamState):
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
        name: Optional[str] = None
    ):
        super().__init__(value, name=name)

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


def assign_state_values(
    states: Sequence[bst.State],
    state_values: Sequence[PyTree],
    write: bool = True
):
    """
    Assign values to the states.

    Args:
      states: The states to be assigned.
      state_values: The values to be assigned.
      write: Whether to write the values to the states. If False, the values will be restored.
    """
    if write:
        for st, val in zip(states, state_values):
            st.value = val
    else:
        for st, val in zip(states, state_values):
            st.restore_value(val)


def assign_dict_state_values(
    states: Dict[Path, bst.State],
    state_values: Dict[Path, PyTree],
    write: bool = True
):
    """
    Assign values to the states.

    Args:
      states: The states to be assigned.
      state_values: The values to be assigned.
      write: Whether to write the values to the states. If False, the values will be restored.
    """
    if set(states.keys()) != set(state_values.keys()):
        raise ValueError('The keys of states and state_values must be the same.')

    if write:
        for key in states.keys():
            states[key].value = state_values[key]
    else:
        for key in states.keys():
            states[key].restore_value(state_values[key])


def assign_state_values_v2(
    states: Dict[Hashable, bst.State],
    state_values: Dict[Hashable, PyTree],
    write: bool = True
):
    """
    Assign values to the states.

    Args:
      states: The states to be assigned.
      state_values: The values to be assigned.
      write: Whether to write the values to the states. If False, the values will be restored.
    """
    assert set(states.keys()) == set(state_values.keys()), 'The keys of states and state_values must be the same.'

    if write:
        for key in states.keys():
            states[key].value = state_values[key]
    else:
        for key in states.keys():
            states[key].restore_value(state_values[key])


def split_states(
    states: Sequence[bst.State]
) -> Tuple[List[bst.ParamState], List[ETraceState], List[bst.State]]:
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
        if isinstance(st, ETraceState):  # etrace hidden variables
            hidden_states.append(st)
        elif isinstance(st, bst.ParamState):  # including all weight states, ParamState, ETraceParam
            param_states.append(st)
        else:
            other_states.append(st)
    return param_states, hidden_states, other_states


def split_states_v2(
    states: Sequence[bst.State]
) -> Tuple[List[ETraceParam], List[ETraceState], List[bst.ParamState], List[bst.State]]:
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
        if isinstance(st, ETraceState):
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


def sequence_split_state_values(
    states: Sequence[bst.State],
    state_values: List[PyTree],
    include_weight: bool = True
) -> (
    Tuple[Sequence[PyTree], Sequence[PyTree], Sequence[PyTree]]
    |
    Tuple[Sequence[PyTree], Sequence[PyTree]]
):
    """
    Split the state values into the weight values, the hidden values, and the other state values.

    The weight values are the values of the ``braincore.ParamState`` states (including ``ETraceParam``).
    The hidden values are the values of the ``ETraceState`` states.
    The other state values are the values of the other states.

    Parameters:
    -----------
    states: Sequence[bst.State]
      The states of the model.
    state_values: List[PyTree]
      The values of the states.
    include_weight: bool
      Whether to include the weight values.

    Returns:
    --------
    The weight values, the hidden values, and the other state values.

    Examples:
    ---------
    >>> sequence_split_state_values(states, state_values)
    (weight_vals, hidden_vals, other_vals)

    >>> sequence_split_state_values(states, state_values, include_weight=False)
    (hidden_vals, other_vals)
    """
    if include_weight:
        weight_vals, hidden_vals, other_vals = [], [], []
        for st, val in zip(states, state_values):
            if isinstance(st, bst.ParamState):
                weight_vals.append(val)
            elif isinstance(st, ETraceState):
                hidden_vals.append(val)
            else:
                other_vals.append(val)
        return weight_vals, hidden_vals, other_vals
    else:
        hidden_vals, other_vals = [], []
        for st, val in zip(states, state_values):
            if isinstance(st, bst.ParamState):
                pass
            elif isinstance(st, ETraceState):
                hidden_vals.append(val)
            else:
                other_vals.append(val)
        return hidden_vals, other_vals



def dict_split_state_values(
    states: Dict[Path, bst.State],
    state_values: Dict[Path, PyTree],
)-> Tuple[WeightVals, HiddenVals, StateVals]:
    weight_vals = bst.util.FlattedDict()
    hidden_vals = bst.util.FlattedDict()
    other_vals = bst.util.FlattedDict()
    for path, state in states.items():
        val = state_values[path]
        if isinstance(state, bst.ParamState):
            weight_vals[path] = val
        elif isinstance(state, ETraceState):
            hidden_vals[path] = val
        else:
            other_vals[path] = val
    return weight_vals, hidden_vals, other_vals

def split_dict_states_v1(
    states: Dict[Path, bst.State]
) -> Tuple[
    Dict[Path, ETraceState],
    Dict[Path, bst.ParamState],
    Dict[Path, bst.State]
]:
    """
    Split the states into weight states, hidden states, and other states.

    .. note::

        This function is important since it determines what ParamState should be
        trained with the eligibility trace and what should not.

    Args:
      states: The states to be split.

    Returns:
      hidden_states: The hidden states.
      param_states: The other kinds of parameter states.
      other_states: The other states.
    """
    hidden_states = bst.util.FlattedDict()
    param_states = bst.util.FlattedDict()
    other_states = bst.util.FlattedDict()
    for key, st in states.items():
        if isinstance(st, ETraceState):
            hidden_states[key] = st
        elif isinstance(st, bst.ParamState):
            # The ParamState which is not an ETraceParam,
            # should be treated as a normal parameter state
            # and be trained with spatial gradients only
            param_states[key] = st
        else:
            other_states[key] = st
    return hidden_states, param_states, other_states


def split_dict_states_v2(
    states: Dict[Path, bst.State]
) -> Tuple[
    Dict[Path, ETraceParam],
    Dict[Path, ETraceState],
    Dict[Path, bst.ParamState],
    Dict[Path, bst.State]
]:
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
    etrace_param_states = bst.util.FlattedDict()
    hidden_states = bst.util.FlattedDict()
    param_states = bst.util.FlattedDict()
    other_states = bst.util.FlattedDict()
    for key, st in states.items():
        if isinstance(st, ETraceState):
            hidden_states[key] = st
        elif isinstance(st, ETraceParam):
            if st.is_not_etrace:
                # The ETraceParam is set to "is_not_etrace" since
                # no hidden state is associated with it,
                # so it should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states[key] = st
            else:
                etrace_param_states[key] = st
        else:
            if isinstance(st, bst.ParamState):
                # The ParamState which is not an ETraceParam,
                # should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states[key] = st
            else:
                other_states[key] = st
    return etrace_param_states, hidden_states, param_states, other_states
