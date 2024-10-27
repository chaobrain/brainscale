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

from typing import Any, Dict, Sequence, Union, FrozenSet, List

import brainstate as bst
import jax

__all__ = [
  'PyTree', 'StateID', 'WeightID', 'Size', 'Axis', 'Axes',
  'Inputs', 'Outputs',
  'HiddenVals', 'StateVals', 'WeightVals', 'ETraceVals',
  'HiddenInVar', 'HiddenOutVar',
  'dG_Inputs', 'dG_Weight', 'dG_Hidden', 'dG_State',
  'ArrayLike', 'DType', 'DTypeLike', 'WeightXVar', 'WeightYVar',
  'WeightXs', 'WeightDfs', 'TempData', 'Current', 'Conductance', 'Spike',
  'Hid2WeightJacobian', 'Hid2HidJacobian', 'Hid2HidDiagJacobian',
]

ArrayLike = bst.typing.ArrayLike
DType = bst.typing.DType
DTypeLike = bst.typing.DTypeLike

# --- types --- #
PyTree = Any
StateID = int
WeightID = int
Size = bst.typing.Size
Axis = int
Axes = Union[int, Sequence[int]]

# --- inputs and outputs --- #
Inputs = PyTree
Outputs = PyTree

# --- state values --- #
HiddenVals = Sequence[PyTree]
StateVals = Sequence[PyTree]
WeightVals = Sequence[PyTree]
ETraceVals = PyTree
HiddenOutVar = jax.core.Var
HiddenInVar = jax.core.Var

# --- gradients --- #
dG_Inputs = PyTree  # gradients of inputs
dG_Weight = Sequence[PyTree]  # gradients of weights
dG_Hidden = Sequence[PyTree]  # gradients of hidden states
dG_State = Sequence[PyTree]  # gradients of other states

# --- data --- #
WeightXVar = jax.core.Var
WeightYVar = jax.core.Var
WeightXs = Dict[jax.core.Var, jax.Array]
WeightDfs = Dict[jax.core.Var, jax.Array]
TempData = Dict[jax.core.Var, jax.Array]
Current = ArrayLike  # the synaptic current
Conductance = ArrayLike  # the synaptic conductance
Spike = ArrayLike  # the spike signal
# the diagonal Jacobian of the hidden-to-hidden function
Hid2HidDiagJacobian = Dict[FrozenSet[HiddenOutVar], Dict[HiddenOutVar, List[jax.Array]]]
Hid2WeightJacobian = Any
Hid2HidJacobian = Any
