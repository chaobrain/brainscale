# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Sequence, Tuple, List, Hashable, Dict

import brainstate as bst

from ._etrace_concepts import ETraceState, ETraceParam
from ._typing import Path, WeightVals, HiddenVals, StateVals


def assign_state_values(
    states: Sequence[bst.State],
    state_values: Sequence[bst.typing.PyTree],
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
    state_values: Dict[Path, bst.typing.PyTree],
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
    state_values: Dict[Hashable, bst.typing.PyTree],
    write: bool = True
):
    """
    Assign values to the states.

    Args:
      states: The states to be assigned.
      state_values: The values to be assigned.
      write: Whether to write the values to the states. If False, the values will be restored.
    """
    assert set(states.keys()) == set(state_values.keys()), (
        f'The keys of states and state_values must be '
        f'the same. Got: \n '
        f'{states.keys()} \n '
        f'{state_values.keys()}'
    )

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
) -> Tuple[
    List[ETraceParam],
    List[ETraceState],
    List[bst.ParamState],
    List[bst.State]
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
    etrace_param_states, hidden_states, param_states, other_states = [], [], [], []
    for st in states:
        if isinstance(st, ETraceState):
            hidden_states.append(st)
        elif isinstance(st, ETraceParam):
            if st.is_etrace:
                etrace_param_states.append(st)
            else:
                # The ETraceParam is not set to "is_etrace" since
                # no hidden state is associated with it,
                # so it should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states.append(st)

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
    state_values: List[bst.typing.PyTree],
    include_weight: bool = True
) -> (
    Tuple[Sequence[bst.typing.PyTree], Sequence[bst.typing.PyTree], Sequence[bst.typing.PyTree]]
    |
    Tuple[Sequence[bst.typing.PyTree], Sequence[bst.typing.PyTree]]
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
    state_values: Dict[Path, bst.typing.PyTree],
) -> Tuple[WeightVals, HiddenVals, StateVals]:
    weight_vals = dict()
    hidden_vals = dict()
    other_vals = dict()
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
    hidden_states = dict()
    param_states = dict()
    other_states = dict()
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
    etrace_param_states = dict()
    hidden_states = dict()
    param_states = dict()
    other_states = dict()
    for key, st in states.items():
        if isinstance(st, ETraceState):
            hidden_states[key] = st
        elif isinstance(st, ETraceParam):
            if st.is_etrace:
                etrace_param_states[key] = st
            else:
                # The ETraceParam is not set to "is_etrace" since
                # no hidden state is associated with it,
                # so it should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states[key] = st
        else:
            if isinstance(st, bst.ParamState):
                # The ParamState which is not an ETraceParam,
                # should be treated as a normal parameter state
                # and be trained with spatial gradients only
                param_states[key] = st
            else:
                other_states[key] = st
    return etrace_param_states, hidden_states, param_states, other_states
