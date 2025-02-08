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

from typing import Any, Dict, Sequence, Union, FrozenSet, List, Tuple

import brainstate as bst
import jax

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Var
else:
    from jax.extend.core import Var

ArrayLike = bst.typing.ArrayLike
DType = bst.typing.DType
DTypeLike = bst.typing.DTypeLike

# --- types --- #
PyTree = bst.typing.PyTree
StateID = int
WeightID = int
Size = bst.typing.Size
Axis = int
Axes = Union[int, Sequence[int]]
Path = Tuple[str, ...]

# --- inputs and outputs --- #
Inputs = PyTree
Outputs = PyTree

# --- state values --- #
HiddenVals = Dict[Path, PyTree]
StateVals = Dict[Path, PyTree]
WeightVals = Dict[Path, PyTree]
ETraceVals = Dict[Path, PyTree]

HiddenOutVar = Var
HiddenInVar = Var

# --- gradients --- #
dG_Inputs = PyTree  # gradients of inputs
dG_Weight = Sequence[PyTree]  # gradients of weights
dG_Hidden = Sequence[PyTree]  # gradients of hidden states
dG_State = Sequence[PyTree]  # gradients of other states

HiddenGroupName = str
ETraceX_Key = Var
ETraceY_Key = Var
ETraceDF_Key = Tuple[Var, HiddenGroupName]

_WeightPath = Path
_HiddenPath = Path
ETraceWG_Key = Tuple[_WeightPath, ETraceY_Key, HiddenGroupName]

HidHidJac_Key = Tuple[Path, Path]

# --- data --- #
WeightXVar = Var
WeightYVar = Var
WeightXs = Dict[Var, jax.Array]
WeightDfs = Dict[Var, jax.Array]
TempData = Dict[Var, jax.Array]
Current = ArrayLike  # the synaptic current
Conductance = ArrayLike  # the synaptic conductance
Spike = ArrayLike  # the spike signal
# the diagonal Jacobian of the hidden-to-hidden function
Hid2HidDiagJacobian = Dict[FrozenSet[HiddenOutVar], Dict[HiddenOutVar, List[jax.Array]]]
Hid2WeightJacobian = Tuple[
    Dict[ETraceX_Key, jax.Array],
    Dict[ETraceDF_Key, jax.Array]
]
Hid2HidJacobian = Dict[HidHidJac_Key, jax.Array]
HidGroupJacobian = Sequence[jax.Array]
