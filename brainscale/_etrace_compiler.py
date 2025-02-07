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

from typing import NamedTuple, List, Sequence, Tuple

import brainstate as bst
import jax.core

from ._etrace_concepts import (
    ETraceParam,
    ETraceState,
)
from ._typing import (
    PyTree,
    WeightXVar,
    WeightYVar,
    HiddenInVar,
    HiddenOutVar,
    Path
)

if jax.__version_info__ < (0, 4, 38):
    from jax.core import Var, Jaxpr, ClosedJaxpr
else:
    from jax.extend.core import Var, Jaxpr, ClosedJaxpr

__all__ = [
    'HiddenGroup',
    'HiddenParamOpRelation',
    'CompiledGraph',
]


class HiddenGroup(NamedTuple):
    r"""
    The data structure for recording the hidden-to-hidden relation.

    The following fields are included:

    - hidden_paths: the path to each hidden state

    This relation is used for computing the hidden-to-hidden state transitions::

        h_{t+1} = f(h_t, x_t)

    where ``h_t`` is the hidden state defined in ``hidden_vars``, ``x_t`` is the input at time ``t``
    defined in ``input_vars``, and ``f`` is the hidden state transition function which is defined
    in ``jaxpr``.

    """

    index: int  # the index of the hidden group

    hidden_paths: List[Path]  # the hidden state paths
    hidden_states: List[ETraceState]  # the hidden states

    # the jax Var at the last time step
    hidden_invars: List[HiddenInVar]  # the input hidden states

    # the jax Var at the current time step
    hidden_outvars: List[HiddenOutVar]  # the output hidden states

    # the jaxpr for computing hidden state transitions
    #
    # h_1^t, h_2^t, ... = f(h_1^{t-1}, h_2^{t-1}, ..., x)
    #
    transition_jaxpr: Jaxpr

    # the other input variables for transition_jaxpr evaluation
    transition_jaxpr_invars: List[Var]

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        The shape of each state variable.
        """
        return self.hidden_states[0].varshape

    @property
    def num_state(self) -> int:
        """
        The number of hidden states.
        """
        return sum([st.num_state for st in self.hidden_states])

    def state_transition(
        self,
        hidden_vals: Sequence[jax.Array],
        input_vals: PyTree,
    ) -> List[jax.Array]:
        """
        Computing the hidden state transitions $h_1^t, h_2^t, \cdots = f(h_1^{t-1}, h_2^{t-1}, \cdots, x^t)$.

        Args:
            hidden_vals: The old hidden state value.
            input_vals: The input values.

        Returns:
            The new hidden state values.
        """
        return jax.core.eval_jaxpr(
            self.transition_jaxpr,
            input_vals,
            hidden_vals
        )


class HiddenParamOpRelation(NamedTuple):
    """
    The data structure for recording the weight, operator, and hidden relationship.

    This is the most important data structure for the eligibility trace compiler.
    It summarizes the parameter, operator, and hidden state relationship, which is used for computing
    the weight spatial gradients and the hidden state Jacobian.

    The following fields are included:

    - ``weight``: the instance of ``ETraceParam``.
    - ``path``: the path to the weight.
    - ``op_jaxpr``: the jaxpr for the weight operation, instance of ``Jaxpr``.
    - ``x``: the jax Var for the weight input. It can be None if the weight is :py:class:`ElemWiseParam`
    - ``y``: the jax Var for the weight output.
    - ``jaxpr_y2hid``: the jaxpr to evaluate y --> eligibility trace hidden states.
    - ``hidden_groups``: the hidden groups that the weight is associated with.

    .. note::

        :py:class:`HiddenParamOpRelation` is uniquely identified by the ``y`` variable.

    """

    weight: ETraceParam  # the weight itself
    path: Path  # the path to the weight
    x: WeightXVar | None  # the input jax var, None if the weight is ElemWiseParam
    y: WeightYVar  # the output jax var
    y_to_hidden_groups_jaxpr: List[Jaxpr]  # the jaxpr for computing y --> hidden groups
    hidden_groups: List[HiddenGroup]

    # hidden_path_to_transition: Dict[Path, HiddenTransition]  # consider to remove
    # hidden_paths: List[Path]  # consider to remove
    # op_jaxpr: Jaxpr  # consider to remove
    # jaxpr_y2hid: Jaxpr  # consider to remove


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
