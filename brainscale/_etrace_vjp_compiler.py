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
# ==============================================================================
#
# Author: Chaoming Wang <chao.brain@qq.com>
# Copyright: 2024, Chaoming Wang
# Date: 2024-04-03
#
# Refinement History:
#   [2024-04-03] Created
#   [2024-04-06] Added the traceback information for the error messages.
#   [2024-04-16] Changed the "op" in the "HiddenWeightOpTracer" to "JaxprEqn".
#                Added the support for the "pjit" operator.
#   [2024-05] Add the support for vjp_time == 't_minus_1'
#   [2024-06] Conditionally support control flows, including `scan`, `while`, and `cond`
#   [2024-09] version 0.0.2
#   [2024-11-22] compatible with `brainstate>=0.1.0` (#17)
#   [2024-11-23] Add the support for vjp_time_ahead > 1, it can combine the
#                advantage of etrace learning and backpropagation through time.
#   [2024-11-26] version 0.0.3, a complete new revision for better model debugging.
#   [2024-12-05] change the ETraceWeight to NonETraceWeight if the hidden states are not found;
#                remove the connected hidden states when y=x@w is not shape broadcastable with the hidden states.
#   [2024-12-09] small updates, related to the key items in "CompiledVjpGraph"
#   [2025-02-06]
#       - [x] unify model retrieved states (brainstate.graph.states)
#             and compiled states (brainstate.compile.StatefulFunction)
#       - [x] add the support for the "ETraceGroupState" and "ETraceTreeState"
#       - [x] add the support for the "ElemWiseParam"
#       - [x] split into "_etrace_compiler.py" and "_etrace_vjp_compiler.py"
#
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Dict, Sequence

import brainstate as bst
import jax.core

from ._etrace_compiler_graph import (
    CompiledGraph,
)
from ._etrace_compiler_hid_param_op import (
    find_hidden_param_op_relations_from_jaxpr,
    HiddenParamOpRelation,
)
from ._etrace_compiler_hidden_group import (
    find_hidden_groups_from_jaxpr,
    HiddenGroup,
)
from ._etrace_compiler_hidden_pertubation import (
    add_hidden_perturbation_in_jaxpr,
)
from ._etrace_compiler_util import (
    abstractify_model,
)
from ._etrace_concepts import (
    ETraceParam,
    ETraceState,
)
from ._misc import _remove_quantity
from ._state_managment import (
    sequence_split_state_values,
)
from ._typing import (
    StateID,
    Path
)

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Var, Jaxpr, ClosedJaxpr
else:
    from jax.extend.core import Var, Jaxpr, ClosedJaxpr

__all__ = [
    'CompiledVjpGraph',
]


def order_hidden_group_index(
    hidden_groups: Sequence[HiddenGroup],
):
    for i, group in enumerate(hidden_groups):
        group.index = i


class CompiledVjpGraph(CompiledGraph):
    """
    The compiled graph for the eligibility trace.

    The following fields are included:

    - augmented_jaxpr: the jaxpr that return necessary intermediate variables
    - jaxpr_perturb_hidden: the jaxpr the add hidden perturbation
    - compiled_model_states: the states of the stateful function
    - stateful_fn_outtree: the output tree of the stateful function
    - hidden_param_op_relations: the hidden-to-weight relation
    - hid_invar_to_path: the mapping from the hidden input variable to the hidden state path
    - hid_outvar_to_path: the mapping from the hidden output variable to the hidden state path
    - hidden_groups: the hidden groups, a sequence of :class:`HiddenGroup` instances
    - hid_path_to_hid_group: the mapping from the hidden state path to the associated hidden group
    - out_hidden_jaxvars: the output hidden jax variables
    - out_wx_jaxvars: the output weight x jax variables
    - out_all_jaxvars: the output all jax variables
    - out_state_jaxvars: the output state jax variables
    - num_out: the number of outputs

    """
    augmented_jaxpr: ClosedJaxpr  # the jaxpr which returns necessary intermediate variables
    jaxpr_perturb_hidden: ClosedJaxpr  # the jaxpr which adds the hidden perturbation
    retrieved_model_states: bst.util.FlattedDict[Path, bst.State]  # the states retrieved by ``module.states()``
    compiled_model_states: Sequence[bst.State]  # the states compiled by the stateful function
    stateful_fn_outtree: jax.tree_util.PyTreeDef
    hidden_groups: Sequence[HiddenGroup]
    hid_path_to_hid_group: Dict[Path, HiddenGroup]
    hidden_param_op_relations: Sequence[HiddenParamOpRelation]
    hid_invar_to_path: Dict[Var, Path]
    hid_outvar_to_path: Dict[Var, Path]
    out_hidden_jaxvars: List[Var]
    out_wx_jaxvars: List[Var]
    out_all_jaxvars: List[Var]
    out_state_jaxvars: List[Var]
    num_out: int


def compile_vjp_graph(
    model: bst.nn.Module,
    compile_to_multi_step: bool,
    *model_args,
    **model_kwargs
) -> CompiledVjpGraph:
    """
    Building the eligibility trace graph for the model according to the given inputs.

    This is the most important method for the eligibility trace graph. It builds the
    graph for the model, which is used for computing the weight spatial gradients and
    the hidden state Jacobian.

    Args:
        model: The model for the eligibility trace.
        compile_to_multi_step: Whether the model is compiled to the ``multi-step``. bool.
        model_args: The model arguments.
        model_kwargs: The model keyword arguments.
    """

    (
        stateful_model,
        model_retrieved_states
    ) = abstractify_model(
        model=model,
        *model_args,
        **model_kwargs
    )

    # stateful model cache key
    cache_key = stateful_model.get_arg_cache_key(*model_args, **model_kwargs)

    # -- states -- #
    compiled_states = stateful_model.get_states(cache_key)

    # -- states information -- #
    path_to_state: Dict[Path, bst.State] = {
        path: state
        for path, state in model_retrieved_states.items()
    }
    state_id_to_path: Dict[StateID, Path] = {
        id(state): path
        for path, state in model_retrieved_states.items()
    }

    # -- jaxpr -- #
    closed_jaxpr = stateful_model.get_jaxpr(cache_key)
    jaxpr = closed_jaxpr.jaxpr

    # -- finding the corresponding in/out vars of etrace states and weights -- #
    out_shapes = stateful_model.get_out_shapes(cache_key)[0]
    state_vals = [state.value for state in compiled_states]
    in_avals, _ = jax.tree.flatten((model_args, model_kwargs))
    out_avals, _ = jax.tree.flatten(out_shapes)
    num_in = len(in_avals)
    num_out = len(out_avals)
    state_avals, state_tree = jax.tree.flatten(state_vals)
    assert len(jaxpr.invars) == len(in_avals) + len(state_avals)
    assert len(jaxpr.outvars) == num_out + len(state_avals)
    invars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.invars[num_in:])
    outvars_with_state_tree = jax.tree.unflatten(state_tree, jaxpr.outvars[num_out:])

    # remove the quantity from the invars and outvars
    invars_with_state_tree = _remove_quantity(invars_with_state_tree)
    outvars_with_state_tree = _remove_quantity(outvars_with_state_tree)

    # -- checking weights as invar -- #
    weight_path_to_invar = {
        state_id_to_path[id(st)]: jax.tree.leaves(invar)
        for invar, st in zip(invars_with_state_tree, compiled_states)
        if isinstance(st, ETraceParam)
    }
    hidden_path_to_invar = {  # one-to-many mapping
        state_id_to_path[id(st)]: invar  # ETraceState only contains one Array, "invar" is the jaxpr var
        for invar, st in zip(invars_with_state_tree, compiled_states)
        if isinstance(st, ETraceState)
    }
    invar_to_hidden_path = {
        invar: path
        for path, invar in hidden_path_to_invar.items()
    }
    invar_to_weight_path = {  # many-to-one mapping
        v: k
        for k, vs in weight_path_to_invar.items()
        for v in vs
    }

    # -- checking states as outvar -- #
    hidden_path_to_outvar = {  # one-to-one mapping
        state_id_to_path[id(st)]: outvar  # ETraceState only contains one Array, "outvar" is the jaxpr var
        for outvar, st in zip(outvars_with_state_tree, compiled_states)
        if isinstance(st, ETraceState)
    }
    outvar_to_hidden_path = {  # one-to-one mapping
        v: state_id
        for state_id, v in hidden_path_to_outvar.items()
    }
    hidden_outvar_to_invar = {
        outvar: hidden_path_to_invar[hid]
        for hid, outvar in hidden_path_to_outvar.items()
    }

    # -- evaluating the relationship for hidden-to-hidden -- #
    weight_invars = set([v for vs in weight_path_to_invar.values() for v in vs])
    (
        hidden_groups,
        hid_path_to_group,
    ) = find_hidden_groups_from_jaxpr(
        jaxpr=jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_invars=weight_invars,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
        path_to_state=path_to_state,
    )

    # order the hidden group index
    order_hidden_group_index(hidden_groups)

    # -- evaluating the jaxpr for (hidden, weight, op) relationships -- #
    hidden_param_op_relations = find_hidden_param_op_relations_from_jaxpr(
        jaxpr=jaxpr,
        hidden_outvar_to_invar=hidden_outvar_to_invar,
        weight_path_to_vars=weight_path_to_invar,
        invar_to_weight_path=invar_to_weight_path,
        path_to_state=path_to_state,
        hid_path_to_group=hid_path_to_group,
        hid_path_to_transition=hid_path_to_transition,
        invar_to_hidden_path=invar_to_hidden_path,
        outvar_to_hidden_path=outvar_to_hidden_path,
    )

    # --- Collect the Var needed to compute the weight spatial gradients --- #
    # ---      Rewrite the jaxpr for computing the needed variables      --- #

    # all states jaxpr var
    out_state_jaxvars = list(jaxpr.outvars[num_out:])
    (
        weight_jaxvar_tree,
        hidden_jaxvar,
        other_state_jaxvar_tree
    ) = sequence_split_state_values(compiled_states, outvars_with_state_tree)

    # all jaxpr var of ETraceState, one etrace to one Array
    # All ETrace Hidden state
    out_hidden_jaxvars = list(hidden_jaxvar)

    # all weight x
    out_wx_jaxvars = list(
        set([relation.x for relation in hidden_param_op_relations])
    )

    # all y-to-hidden vars
    out_wy2hid_jaxvars = list(
        set(
            [v for relation in hidden_param_op_relations
             for v in (relation.jaxpr_y2hid.invars +
                       relation.jaxpr_y2hid.constvars)]
        )
    )

    # hidden-hidden transition vars
    hid2hid_jaxvars = set()
    for group in hidden_groups:
        hid2hid_jaxvars.update([v for v in group.hidden_invars])
    for transition in hid_path_to_transition.values():
        hid2hid_jaxvars.update([v for v in transition.other_invars])

    # all outvars
    all_outvars = list(
        set(
            out_state_jaxvars +  # all state variables
            out_wx_jaxvars +  # all weight x
            out_wy2hid_jaxvars +  # all y-to-hidden invars
            list(hid2hid_jaxvars)  # all hidden-hidden transition vars
        )
    )
    out_all_jaxvars = jaxpr.outvars[:num_out] + all_outvars

    # Rewrite jaxpr to return all necessary variables, including
    #
    #   1. the original function outputs
    #   2. the hidden states
    #   3. the weight x   ===>  for computing the weight spatial gradients
    #   4. the y-to-hidden variables   ===>  for computing the weight spatial gradients
    #   5. the hidden-hidden transition variables   ===>  for computing the hidden-hidden jacobian
    #
    jaxpr = Jaxpr(
        constvars=list(jaxpr.constvars),
        invars=list(jaxpr.invars),
        outvars=list(out_all_jaxvars),
        eqns=list(jaxpr.eqns),
        effects=jaxpr.effects,
        debug_info=jaxpr.debug_info,
    )
    augmented_jaxpr = ClosedJaxpr(jaxpr, closed_jaxpr.consts)

    if compile_to_multi_step:
        jaxpr_with_hidden_perturb = None

    else:
        # ---               add perturbations to the hidden states                  --- #
        # --- new jaxpr with hidden state perturbations for computing the residuals --- #

        jaxpr_with_hidden_perturb = add_hidden_perturbation_in_jaxpr(
            closed_jaxpr=augmented_jaxpr,
            weight_invars=weight_invars,
            invar_to_hidden_path=invar_to_hidden_path,
            outvar_to_hidden_path=outvar_to_hidden_path,
            hidden_outvar_to_invar=hidden_outvar_to_invar,
        )

    return CompiledVjpGraph(
        augmented_jaxpr=augmented_jaxpr,
        jaxpr_perturb_hidden=jaxpr_with_hidden_perturb,
        retrieved_model_states=model_retrieved_states,
        compiled_model_states=stateful_model.get_states(cache_key),
        stateful_fn_outtree=stateful_model.get_out_treedef(cache_key),
        hidden_groups=hidden_groups,
        hid_path_to_hid_group=hid_path_to_group,
        hidden_param_op_relations=hidden_param_op_relations,
        hid_invar_to_path=invar_to_hidden_path,
        hid_outvar_to_path=outvar_to_hidden_path,
        out_hidden_jaxvars=out_hidden_jaxvars,
        out_wx_jaxvars=out_wx_jaxvars,
        out_all_jaxvars=out_all_jaxvars,
        out_state_jaxvars=out_state_jaxvars,
        num_out=num_out,
    )
