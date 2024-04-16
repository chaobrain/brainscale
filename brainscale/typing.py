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
from typing import Any, Dict, Sequence, Union

import braincore as bc
import jax

__all__ = [
  'PyTree', 'StateID', 'WeightID', 'Size', 'Axis', 'Axes',
  'Inputs', 'Outputs',
  'HiddenVals', 'StateVals', 'WeightVals', 'ETraceVals', 'HiddenVar',
  'dG_Inputs', 'dG_Weight', 'dG_Hidden', 'dG_State',
  'ArrayLike', 'DType', 'DTypeLike',
  'WeightXs', 'WeightDfs', 'TempData', 'Current', 'Conductance', 'Spike',
]

ArrayLike = bc.typing.ArrayLike
DType = bc.typing.DType
DTypeLike = bc.typing.DTypeLike

# --- types --- #
PyTree = Any
StateID = int
WeightID = int
Size = bc.typing.Size
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
HiddenVar = jax.core.Var

# --- gradients --- #
dG_Inputs = PyTree  # gradients of inputs
dG_Weight = Sequence[PyTree]  # gradients of weights
dG_Hidden = Sequence[PyTree]  # gradients of hidden states
dG_State = Sequence[PyTree]  # gradients of other states

# --- data --- #
WeightXVar = jax.core.Var
WeightXs = Dict[jax.core.Var, jax.Array]
WeightDfs = Dict[jax.core.Var, jax.Array]
TempData = Dict[jax.core.Var, jax.Array]
Current = ArrayLike  # the synaptic current
Conductance = ArrayLike  # the synaptic conductance
Spike = ArrayLike  # the spike signal
