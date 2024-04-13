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

from typing import Union, List

import jax


def get_node_label(
    v: Union[jax.core.Var, jax.core.Literal],
    show_avals: bool
) -> str:
  """
  Concatenate a variable name and its type.

  Parameters
  ----------
  v: Var
      Jax variable
  show_avals: bool
      If `True` then the type will be included in the
      node label

  Returns
  -------
  str
  """
  if show_avals:
    return f"{v}: {v.aval.str_short()}"
  else:
    return str(v)


def is_not_primitive(x: jax.core.JaxprEqn) -> bool:
  """
  Test if a JaxprEqn is a primitive.

  Parameters
  ----------
  x: JaxprEqn

  Returns
  -------
  bool
      'True' if not a primitive.
  """
  return x.primitive.name == "pjit"


def contains_non_primitives(eqns: List[jax.core.JaxprEqn]) -> bool:
  """
  Check it the sub-functions of a JaxPR contains only JAX primitives

  Parameters
  ----------
  eqns: List[jax._src.core.JaxprEqn]
      List of JaxprEqns

  Returns
  -------
  bool:
      `True` if any of the sub-eqns are non-primitive
  """
  return any(
    [
      ("jaxpr" in e.params or e.primitive.name in {"cond", "scan", "while"})
      for e in eqns
    ]
  )
