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
# Copyright: 2024, Chaoming Wang
# Date: 2024-04-03
#
# ==============================================================================
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
#   [2024-11] version 0.0.3, a complete new revision for better model debugging.
#
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import (Dict, Tuple)

import brainstate as bst

from ._etrace_compiler_graph import CompiledGraph
from ._typing import (
    Outputs,
    HiddenVals,
    StateVals,
    Hid2WeightJacobian,
    Hid2HidJacobian,
    Path,
)

__all__ = [
    'ETraceGraphExecutor',
]


class ETraceGraphExecutor:
    r"""
    The eligibility trace graph, tracking the relationship between the etrace weights
    :py:class:`ETraceParam`, the etrace variables :py:class:`ETraceState`, and the etrace
    operations :py:class:`ETraceOp`.

    This class is used for computing the weight spatial gradients and the hidden state residuals.
    It is the most foundational data structure for the ETrace algorithms.

    It is important to note that the graph is built no matter whether the model is
    batched or not. This means that this graph can be applied to any kind of models.
    However, the compilation is sensitive to the shape of hidden states.

    Parameters
    ----------
    model: brainstate.nn.Module
        The model to build the eligibility trace graph. The models should only define the one-step behavior.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        model: bst.nn.Module,
    ):
        # The original model
        if not isinstance(model, bst.nn.Module):
            raise TypeError(
                'The model should be an instance of "bst.nn.Module" since '
                'we can extract the program structure from the model for '
                'better debugging.'
            )
        self.model = model

        # the compiled graph
        self._compiled_graph = None
        self._state_id_to_path = None

    @property
    def compiled(self) -> CompiledGraph:
        r"""
        The compiled graph for the model.

        It is the most important data structure for the eligibility trace graph.
        The instance of :py:class:`CompiledVjpGraph`.

        It contains the following attributes:

        - ``out_all_jaxvars``: List[jex.core.Var]  # all outvars except the function returns
        - ``out_state_jaxvars``: List[jex.core.Var]  # the state vars
        - ``out_wx_jaxvars``: List[jex.core.Var]  # the weight x
        - ``hid2hid_jaxvars``: List[jex.core.Var]  # the hidden to hidden vars
        - ``num_out``: int  # the number of function returns


        Hidden states information: input `jex.core.Var` to `ETraceState`, and output `jex.core.Var` to `ETraceState`.

        - ``hidden_invar_to_hidden``: Dict[jex.core.Var, ETraceState]
        - ``hidden_outvar_to_hidden``: Dict[jex.core.Var, ETraceState]


        Intermediate variable relationship: input `jex.core.Var` to `jex.core.Var`,
        and output `jex.core.Var` to `jex.core.Var`.

        - ``hidden_outvar_to_invar``: Dict[jex.core.Var, jex.core.Var]
        - ``hidden_invar_to_outvar``: Dict[jex.core.Var, jex.core.Var]


        The most important data structure for the graph, which implementing
        the relationship between the etrace weights and the etrace states.

        - ``hidden_param_op_relations``: Tuple[HiddenParamOpRelation, ...]


        The relationship between the hidden states, and state transitions.

        - ``hidden_groups``: Sequence[HiddenGroupV1]
        - ``hidden_to_group``: Dict[Path, HiddenGroupV1]  # Path is the hidden state path
        - ``hidden_to_transition``: Dict[Path, Hidden2GroupTransition]  # Path is the hidden state path


        The augmented jaxpr is nearly identical to the original jaxpr, except that
        that it will return all necessary variables, for example, the intermediate
        variables, the hidden states, the weight x, the y-to-hidden variables, and
        the hidden-hidden transition variables.

        - ``augmented_jaxpr``: jax.core.ClosedJaxpr = None


        The revised jaxpr with hidden state perturbations is essential for computing
        the learning signal :math:`\partial L / \partial h`, where :math:`L` is the loss and h is the hidden state.
        It also returns necessary variables. Note, this jaxpr is only needed when the "vjp_time" is "t".

        - ``jaxpr_with_hidden_perturb``: jax.core.ClosedJaxpr = None


        """
        if self._compiled_graph is None:
            raise ValueError('The graph is not compiled yet. Please call ".compile_graph()" first.')
        return self._compiled_graph

    @property
    def states(self) -> bst.util.FlattedDict[Path, bst.State]:
        """
        The states for the model.

        Returns:
            The states for the model.
        """
        return self.compiled.model_retrieved_states

    @property
    def path_to_states(self) -> bst.util.FlattedDict[Path, bst.State]:
        """
        The path to the states.

        Returns:
            The path to the states.
        """
        return self.states

    @property
    def state_id_to_path(self) -> Dict[int, Path]:
        """
        The state id to the path.

        Returns:
            The state id to the path.
        """
        if self._state_id_to_path is None:
            self._state_id_to_path = {id(state): path for path, state in self.states.items()}
        return self._state_id_to_path

    def compile_graph(self, *args):
        r"""
        Building the eligibility trace graph for the model according to the given inputs.

        This is the most important method for the eligibility trace graph. It builds the
        graph for the model, which is used for computing the weight spatial gradients and
        the hidden state Jacobian.

        Args:
            *args: The positional arguments for the model.
        """

        raise NotImplementedError('The method "compile_graph" should be implemented in the subclass.')

    def show_graph(self):
        """
        Showing the graph about the relationship between weight, operator, and hidden states.
        """

        # hidden group
        msg = '===' * 40 + '\n'
        msg += 'The hidden groups are:\n\n'
        hidden_paths = []
        group_mapping = dict()
        for group in self.compiled.hidden_groups:
            msg += f'   Group {group.index}: {group.hidden_paths}\n'
            group_mapping[id(group)] = group.index
            hidden_paths.extend(group.hidden_paths)
        msg += '\n\n'

        # other hidden states
        other_states = []
        short_states = self.states.filter(bst.ShortTermState)
        for i, path in enumerate(short_states.keys()):
            if path not in hidden_paths:
                other_states.append(path)
        if len(other_states):
            msg += 'The dynamic (non-hidden) states are:\n\n'
            for i, path in enumerate(other_states):
                msg += f'   Dynamic state {i}: {path}\n'
            msg += '\n\n'

        # etrace weights
        etratce_weight_paths = set()
        if len(self.compiled.hidden_param_op_relations):
            msg += 'The weight parameters which are associated with the hidden states are:\n\n'
            for i, hp_relation in enumerate(self.compiled.hidden_param_op_relations):
                etratce_weight_paths.add(hp_relation.path)
                group = [group_mapping[id(group)] for group in hp_relation.hidden_groups]
                if len(group) == 1:
                    msg += f'   Weight {i}: {hp_relation.path}  is associated with hidden group {group[0]}\n'
                else:
                    msg += f'   Weight {i}: {hp_relation.path}  is associated with hidden groups {group}\n'
            msg += '\n\n'

        # non etrace weights
        non_etratce_weight_paths = set(self.states.filter(bst.ParamState).keys())
        non_etratce_weight_paths = non_etratce_weight_paths.difference(etratce_weight_paths)
        if len(non_etratce_weight_paths):
            msg += 'The non-etrace weight parameters are:\n\n'
            for i, path in enumerate(non_etratce_weight_paths):
                msg += f'   Weight {i}: {path}\n'
            msg += '\n\n'

        print(msg)

    def solve_h2w_h2h_jacobian(
        self,
        *args,
    ) -> Tuple[
        Outputs,
        HiddenVals,
        StateVals,
        Hid2WeightJacobian,
        Hid2HidJacobian,
    ]:
        r"""
        Solving the hidden-to-weight and hidden-to-hidden Jacobian according to the given inputs and parameters.

        This function is typically used for computing the forward propagation of hidden-to-weight Jacobian.

        Now we mathematically define what computations are done in this function.

        For the state transition function $y, h^t = f(h^{t-1}, \theta, x)$, this function aims to solve:

        1. The function output: $y$
        2. The updated hidden states: $h^t$
        3. The Jacobian matrix of hidden-to-weight, i.e., $\partial h^t / \partial \theta^t$.
        2. The Jacobian matrix of hidden-to-hidden, i.e., $\partial h^t / \partial h^{t-1}$.

        Args:
            *args: The positional arguments for the model.

        Returns:
            The outputs, hidden states, other states, and the spatial gradients of the weights.
        """
        raise NotImplementedError('The method "solve_h2w_h2h_jacobian" should be '
                                  'implemented in the subclass.')

    def solve_h2w_h2h_jacobian_and_l2h_vjp(
        self, *args,
    ) -> Tuple[Outputs, HiddenVals, StateVals, Hid2WeightJacobian, Hid2HidJacobian, ...]:
        r"""
        Solving the hidden-to-weight and hidden-to-hidden Jacobian and the VJP transformed loss-to-hidden
        gradients according to the given inputs.

        This function is typically used for computing both the forward propagation of hidden-to-weight Jacobian
        and the loss-to-hidden gradients at the current time-step.

        Particularly, this function aims to solve:

        1. The Jacobian matrix of hidden-to-weight. That is,
           :math:`\partial h / \partial w`, where :math:`h` is the hidden state and :math:`w` is the weight.
        2. The Jacobian matrix of hidden-to-hidden. That is,
           :math:`\partial h / \partial h`, where :math:`h` is the hidden state.
        3. The partial gradients of the loss with respect to the hidden states.
           That is, :math:`\partial L / \partial h`, where :math:`L` is the loss and :math:`h` is the hidden state.

        Args:
          *args: The positional arguments for the model.

        Returns:
          The outputs, hidden states, other states, the spatial gradients of the weights, and the residuals.
        """
        raise NotImplementedError('The method "solve_h2w_h2h_jacobian_and_l2h_vjp" '
                                  'should be implemented in the subclass.')
