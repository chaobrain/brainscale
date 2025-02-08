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

from typing import Dict, Sequence, NamedTuple, Tuple

import brainstate as bst

from ._etrace_compiler_base import (
    extract_model_info,
    ModelInfo,
)
from ._etrace_compiler_hid_param_op import (
    find_hidden_param_op_relations_from_minfo,
    HiddenParamOpRelation,
)
from ._etrace_compiler_hidden_group import (
    find_hidden_groups_from_minfo,
    HiddenGroup,
)
from ._etrace_compiler_hidden_pertubation import (
    add_hidden_perturbation_from_minfo,
    HiddenPerturbation,
)
from ._typing import Path


def order_hidden_group_index(
    hidden_groups: Sequence[HiddenGroup],
):
    for i, group in enumerate(hidden_groups):
        group.index = i


class CompiledGraph(NamedTuple):
    """
    The compiled graph for the eligibility trace.

    The following fields are included:

    - ``model_info``: The model information, instance of :class:`ModelInfo`.
    - ``hidden_groups``: The hidden groups, sequence of :class:`HiddenGroup`.
    - ``hid_path_to_group``: The mapping from the hidden path to the hidden group :class:`HiddenGroup`.
    - ``hidden_param_op_relations``: The hidden parameter operation relations, sequence of :class:`HiddenParamOpRelation`.
    - ``hidden_perturb``: The hidden perturbation, instance of :class:`HiddenPerturbation`, or None.
    """

    model_info: ModelInfo
    hidden_groups: Sequence[HiddenGroup]
    hid_path_to_group: Dict[Path, HiddenGroup]
    hidden_param_op_relations: Sequence[HiddenParamOpRelation]
    hidden_perturb: HiddenPerturbation | None


CompiledGraph.__module__ = 'brainscale'


def compile_graph(
    model: bst.nn.Module,
    *model_args: Tuple,
    include_hidden_perturb: bool = True,
) -> CompiledGraph:
    """
    Building the eligibility trace graph for the model according to the given inputs.

    This is the most important method for the eligibility trace graph. It builds the
    graph for the model, tracking the relationship between the etrace weights
    :py:class:`ETraceParam`, the etrace sattes :py:class:`ETraceState`, and the etrace
    operations :py:class:`ETraceOp`, which will be used for computing the weight
    spatial gradients, the hidden state Jacobian, and the hidden state-weight Jacobian.

    Args:
        model: The model for the eligibility trace.
        model_args: tuple, The model arguments.
        include_hidden_perturb: bool. Whether to include the hidden perturbation. Default is True.
    """

    assert isinstance(model_args, tuple)
    minfo = extract_model_info(model, *model_args)

    # ---       evaluating the relationship for hidden-to-hidden        --- #
    hidden_groups, hid_path_to_group = find_hidden_groups_from_minfo(minfo)
    order_hidden_group_index(hidden_groups)

    # ---       evaluating the jaxpr for (hidden, param, op) relationships      --- #

    hidden_param_op_relations = find_hidden_param_op_relations_from_minfo(
        minfo=minfo,
        hid_path_to_group=hid_path_to_group,
    )

    # ---      Rewrite the jaxpr for computing the needed variables      --- #

    # Rewrite jaxpr to return all necessary variables, including
    #
    #   1. the original function outputs
    #   2. the hidden states
    #   3. the weight x   ===>  for computing the weight spatial gradients
    #   4. the y-to-hidden variables   ===>  for computing the weight spatial gradients
    #   5. the hidden-hidden transition variables   ===>  for computing the hidden-hidden jacobian
    #

    # all weight x
    out_wx_jaxvars = list(set([
        relation.x for relation in hidden_param_op_relations
        if relation.x is not None
    ]))

    # all y-to-hidden vars
    out_wy2hid_jaxvars = set()
    for relation in hidden_param_op_relations:
        for hpo_relation in relation.y_to_hidden_group_jaxprs:
            out_wy2hid_jaxvars.update(hpo_relation.invars + hpo_relation.constvars)
    out_wy2hid_jaxvars = list(out_wy2hid_jaxvars)

    # hidden-hidden transition vars
    hid2hid_jaxvars = set()
    for group in hidden_groups:
        hid2hid_jaxvars.update(group.hidden_invars)
        hid2hid_jaxvars.update(group.transition_jaxpr_constvars)
    hid2hid_jaxvars = list(hid2hid_jaxvars)

    # all temporary outvars
    temp_outvars = set(
        minfo.jaxpr.outvars[minfo.num_var_out:] +  # all state variables
        out_wx_jaxvars +  # all weight x
        out_wy2hid_jaxvars +  # all y-to-hidden invars
        hid2hid_jaxvars  # all hidden-hidden transition vars
    ).difference(
        minfo.jaxpr.outvars  # exclude the original function outputs
    )

    # rewrite model_info
    minfo = minfo.add_jaxpr_outs(list(temp_outvars))

    # ---               add perturbations to the hidden states                  --- #
    # --- new jaxpr with hidden state perturbations for computing the residuals --- #

    hidden_perturb = add_hidden_perturbation_from_minfo(minfo) if include_hidden_perturb else None

    # ---              return the compiled graph               --- #

    return CompiledGraph(
        model_info=minfo,
        hidden_groups=hidden_groups,
        hid_path_to_group=hid_path_to_group,
        hidden_param_op_relations=hidden_param_op_relations,
        hidden_perturb=hidden_perturb,
    )
