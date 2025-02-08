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

from typing import NamedTuple, Sequence

import brainstate as bst
import jax.core

from ._etrace_compiler_hid_param_op import HiddenParamOpRelation
from ._etrace_compiler_hidden_group import HiddenGroup
from ._typing import Path

if jax.__version_info__ < (0, 4, 38):
    from jax.core import ClosedJaxpr
else:
    from jax.extend.core import ClosedJaxpr

__all__ = [
    'CompiledGraph',
]


class CompiledGraph(NamedTuple):
    """
    The compiled graph for the eligibility trace.

    The following fields are included:

    - augmented_jaxpr: the jaxpr that return necessary intermediate variables
    - jaxpr_perturb_hidden: the jaxpr the add hidden perturbation
    - stateful_fn_states: the states of the stateful function
    - stateful_fn_outtree: the output tree of the stateful function
    - hidden_param_op_relations: the hidden-to-weight relation
    - hid_invar_to_path: the mapping from the hidden input variable to the hidden state path
    - hid_outvar_to_path: the mapping from the hidden output variable to the hidden state path
    - hid_path_to_transition: the mapping from the hidden state path to the hidden state transition
    - out_hidden_jaxvars: the output hidden jax variables
    - out_wx_jaxvars: the output weight x jax variables
    - out_all_jaxvars: the output all jax variables
    - out_state_jaxvars: the output state jax variables
    - num_out: the number of outputs

    """
    augmented_jaxpr: ClosedJaxpr  # the jaxpr which returns necessary intermediate variables
    jaxpr_perturb_hidden: ClosedJaxpr  # the jaxpr which adds the hidden perturbation
    model_retrieved_states: bst.util.FlattedDict[Path, bst.State]  # the states retrieved by ``module.states()``
    stateful_fn_states: Sequence[bst.State]  # the states compiled by the stateful function
    stateful_fn_outtree: jax.tree_util.PyTreeDef
    hidden_groups: Sequence[HiddenGroup]
    hidden_param_op_relations: Sequence[HiddenParamOpRelation]
