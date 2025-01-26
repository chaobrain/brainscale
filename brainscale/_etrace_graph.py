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

from functools import partial
from typing import (Dict, Tuple)

import brainstate as bst
import brainunit as u
import jax.core
import jax.numpy as jnp
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe

from ._etrace_compiler import (HiddenTransition,
                               compile_graph,
                               CompiledGraph)
from ._etrace_concepts import (ETraceState,
                               assign_dict_state_values,
                               dict_split_state_values,
                               split_dict_states_v2)
from ._etrace_input_data import (get_single_step_data,
                                 split_data_types,
                                 merge_data)
from ._typing import (PyTree,
                      TempData,
                      Outputs,
                      HiddenVals,
                      StateVals,
                      ETraceX_Key,
                      ETraceDF_Key,
                      HidHidJac_Key,
                      Hid2WeightJacobian,
                      Hid2HidJacobian)

# TODO
#
# - [x] The visualization of the etrace graph.
# - [ ] Evaluate whether the `df` is the same for different weights.
#       For example,
#
#          h = f(x1 @ w1 + x2 @ w2)
#
#       The `df` for w1 and w2 are the same, although them have the different weight y.

__all__ = [
    'ETraceGraph',
]


@jax.tree_util.register_pytree_node_class
class Residuals:
    """
    The residuals for storing the backward pass data in a VJP function.

    Args:
      jaxpr: The jaxpr for the backward pass.
      in_tree: The input tree structure.
      out_tree: The output tree structure.
      consts: The constants for the backward pass.
    """

    def __init__(self, jaxpr, in_tree, out_tree, consts):
        self.jaxpr = jaxpr
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.consts = consts

    def __iter__(self):
        return iter((self.jaxpr, self.in_tree, self.out_tree, self.consts))

    def tree_flatten(self):
        return self.consts, (self.jaxpr, self.in_tree, self.out_tree)

    @classmethod
    def tree_unflatten(cls, aux, consts):
        jaxpr, in_tree, out_tree = aux
        return cls(jaxpr, in_tree, out_tree, consts)


class ETraceGraph:
    """
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

    def __init__(self, model: bst.nn.Module, vjp_method: str = 'single-step'):
        # the VJP method
        assert vjp_method in ('single-step', 'multi-step'), (
            'The VJP method should be either "single-step" or "multi-step". '
            f'While we got {vjp_method}. '
        )
        self.vjp_method = vjp_method

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

    @property
    def is_single_step(self):
        """
        Whether the VJP method is single-step.

        Returns:
            bool: Whether the VJP method is single-step.
        """
        return self.vjp_method == 'single-step'

    @property
    def is_multi_step(self):
        """
        Whether the VJP method is multi-step.

        Returns:
            bool: Whether the VJP method is multi-step.
        """
        return self.vjp_method == 'multi-step'

    @property
    def compiled(self) -> CompiledGraph:
        r"""
        The compiled graph for the model.

        It is the most important data structure for the eligibility trace graph.
        The instance of :py:class:`CompiledGraph`.

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

        - ``hidden_param_op_relations``: Tuple[WeightOpHiddenRelation, ...]


        The relationship between the hidden states, and state transitions.

        - ``hidden_groups``: Sequence[HiddenGroupV1]
        - ``hidden_to_group``: Dict[Path, HiddenGroupV1]  # Path is the hidden state path
        - ``hidden_to_transition``: Dict[Path, HiddenTransition]  # Path is the hidden state path


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

    def compile_graph(self, *args):
        r"""
        Building the eligibility trace graph for the model according to the given inputs.

        This is the most important method for the eligibility trace graph. It builds the
        graph for the model, which is used for computing the weight spatial gradients and
        the hidden state Jacobian.

        Args:
            *args: The positional arguments for the model.
        """

        # state information
        self.states = bst.graph.states(self.model)
        self.path_to_states = {path: state for path, state in self.states.items()}
        self.state_id_to_path = {id(state): path for path, state in self.states.items()}

        # process the inputs
        args = get_single_step_data(*args)

        # compile the graph
        self._compiled_graph = compile_graph(self.model, self.vjp_method == 'multi-step', *args)

    def show_graph(self):
        """
        Showing the graph about the relationship between weight, operator, and hidden states.
        """

        # hidden group
        msg = '===' * 40 + '\n'
        msg += 'The hidden groups are:\n\n'
        group_mapping = dict()
        for i, group in enumerate(self.compiled.hidden_groups):
            msg += f'   Group {i}: {group.hidden_paths}\n'
            group_mapping[id(group)] = i
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

    def _jaxpr_compute_model(
        self,
        *args,
    ) -> Tuple[PyTree, HiddenVals, StateVals, TempData]:
        """
        Computing the model according to the given inputs and parameters by using the compiled jaxpr.
        """

        # state checking
        old_state_vals = [st.value for st in self.compiled.stateful_fn_states]

        # parameters
        args = jax.tree.flatten((args, old_state_vals))[0]

        # calling the function
        jaxpr_outs = jax.core.eval_jaxpr(
            self.compiled.augmented_jaxpr.jaxpr,
            self.compiled.augmented_jaxpr.consts,
            *args
        )

        # intermediate values
        #
        # "jaxpr_outs[:self.num_out]" corresponds to model original outputs
        # "jaxpr_outs[self.num_out:]" corresponds to extra output in  "augmented_jaxpr"
        temps = {
            v: r for v, r in
            zip(
                self.compiled.out_all_jaxvars[self.compiled.num_out:],
                jaxpr_outs[self.compiled.num_out:]
            )
        }

        #
        # recovery outputs of ``stateful_model``
        state_outs = [temps[v] for v in self.compiled.out_state_jaxvars]
        out, new_state_vals = self.compiled.stateful_fn_outtree.unflatten(
            jaxpr_outs[:self.compiled.num_out] + state_outs
        )

        # state value assignment
        assert len(old_state_vals) == len(new_state_vals), 'State length mismatch.'

        # split the state values
        # Assume that weights are not changed.
        #
        hidden_vals = dict()
        oth_state_vals = dict()
        for st, st_val in zip(self.compiled.stateful_fn_states, new_state_vals):
            if isinstance(st, ETraceState):
                hidden_vals[self.state_id_to_path[id(st)]] = st_val
            elif isinstance(st, bst.ParamState):
                pass
            else:
                if id(st) not in self.state_id_to_path:
                    raise ValueError(f'This state {st} can not be accessed by the model {self.model}. \n'
                                     f'Please assign the state as the attribute of the model.')
                oth_state_vals[self.state_id_to_path[id(st)]] = st_val
        return out, hidden_vals, oth_state_vals, temps

    def _compute_hid2weight_jacobian(
        self,
        intermediate_values: dict
    ) -> Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]:
        """
        Computing the weight x and df values for the spatial gradients.

        Args:
          intermediate_values: The intermediate values of the model.

        Returns:
          The weight x and df values.
        """
        intermediate_values = jax.lax.stop_gradient(intermediate_values)

        # the weight x
        xs = {v: intermediate_values[v] for v in self.compiled.out_wx_jaxvars}

        # the weight df
        dfs = dict()
        for relation in self.compiled.hidden_param_op_relations:
            consts = [intermediate_values[var] for var in relation.jaxpr_y2hid.constvars]
            invars = [intermediate_values[var] for var in relation.jaxpr_y2hid.invars]  # weight y
            assert len(invars) == 1, 'The weight y should be unique.'

            # [ KEY ]
            #
            # # ---- Method 1: using ``backward_pass`` ---- #
            # Assuming the function is linear. One cheap way is to use
            # the backward pass for computing the gradients of the hidden states.
            # For most situations, the ``y --> hidden`` relation is linear. Therefore,
            # we use ``backward_pass`` to compute the ``Df`` while avoids the overhead
            # of computing the forward pass. Otherwise, we should use ``jax.vjp`` instead.
            # Please also see ``jax.linear_transpose()`` for the same purpose.
            #
            # [df] = backward_pass(relation.jaxpr_y2hid, [], True, consts, invars, outvars)
            #
            # # ---- Method 2: using ``jax.vjp`` ---- #
            # However, for general cases, we choose to use ``jax.vjp`` to compute the gradients.
            #
            # # ---- Method 3: using ``jax.jvp`` ---- #
            primals, tangents = jax.jvp(
                lambda x: jax.core.eval_jaxpr(relation.jaxpr_y2hid, consts, x),
                invars,
                [u.math.ones(invars[0].shape, invars[0].dtype)]
            )

            # get the df we want
            for i, hidden_var in enumerate(relation.jaxpr_y2hid.outvars):  # hidden states
                hidden_path = self.compiled.hid_outvar_to_path[hidden_var]
                key = (relation.y, hidden_path)
                if key in dfs:
                    raise ValueError(f'The key should not exist. {key}')
                dfs[key] = tangents[i]

        # all x and df values
        return jax.lax.stop_gradient(xs), jax.lax.stop_gradient(dfs)

    def _compute_hid2hid_jacobian(
        self,
        intermediate_values: dict
    ) -> Dict[HidHidJac_Key, jax.Array]:

        intermediate_values = jax.lax.stop_gradient(intermediate_values)

        hid2hid_jacobian = dict()
        for hid_path, transition in self.compiled.hid_path_to_transition.items():
            transition: HiddenTransition

            #
            # "primals" is the hidden state values at the previous time step
            primals = intermediate_values[transition.hidden_invar]

            #
            # "tangents" is the hidden-to-hidden Jacobian at the previous time step
            tangents = u.math.ones(primals.aval.shape, primals.aval.dtype)

            # JVP gradients, computing:
            #
            # [∂a^t/∂a^t-1, ∂b^t/∂a^t-1, ∂c^t/∂a^t-1, ...]
            #
            other_input_vals = [intermediate_values[v] for v in transition.other_invars]
            fun = partial(transition.state_transition, other_input_vals=other_input_vals)
            _, jvp_grads = jax.jvp(fun, (primals,), (tangents,))  # produce the new hidden, and the JVP gradients

            # store the gradients
            hid_path1 = transition.hidden_path
            for hid_path2, grad_data in zip(transition.connected_hidden_paths, jvp_grads):
                key = (hid_path1, hid_path2)
                assert key not in hid2hid_jacobian, f'The key should not exist. {key}'
                hid2hid_jacobian[key] = grad_data

        return jax.lax.stop_gradient(hid2hid_jacobian)

    def solve_h2w_h2h_jacobian(
        self,
        *args,
    ) -> Tuple[Outputs, HiddenVals, StateVals, Hid2WeightJacobian, Hid2HidJacobian]:
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
        # --- compile the model --- #
        if self.vjp_method != 'multi-step':
            assert self.compiled.jaxpr_perturb_hidden is not None, (
                'The jaxpr_with_hidden_perturb should not be None '
                'when the vjp_time_ahead is 0.'
            )

        assert self.compiled.augmented_jaxpr is not None, (
            'The augmented_jaxpr should not be None '
            'when the vjp_time_ahead > 0.'
        )

        (
            etrace_param_states,
            hidden_states,
            non_etrace_weight_states,
            other_states
        ) = split_dict_states_v2(self.states)

        hidden_vals = {path: st.value for path, st in hidden_states.items()}
        other_vals = {path: st.value for path, st in other_states.items()}
        non_etrace_weight_vals = {path: st.value for path, st in non_etrace_weight_states.items()}
        etrace_weight_vals = {path: st.value for path, st in etrace_param_states.items()}

        # --- call the model --- #

        def scan_fn(carray, single_step_of_multistep_arg):
            args_ = merge_data(tree_def, single_step_of_multistep_arg, args_single_step)

            _hidden_vals, _oth_state_vals = carray
            for path, val in _hidden_vals.items():
                self.path_to_states[path].restore_value(val)
            for path, val in _oth_state_vals.items():
                self.path_to_states[path].restore_value(val)
            for path, val in non_etrace_weight_vals.items():
                self.path_to_states[path].restore_value(val)
            for path, val in etrace_weight_vals.items():
                self.path_to_states[path].restore_value(val)

            (
                out,
                _hidden_vals,
                _oth_state_vals,
                temps
            ) = self._jaxpr_compute_model(*args_)

            # compute the hidden-to-weight Jacobian
            hid2weight_jac = self._compute_hid2weight_jacobian(temps)

            # compute the hidden-to-hidden Jacobian
            hid2hid_jac = self._compute_hid2hid_jacobian(temps)

            return (_hidden_vals, _oth_state_vals), (out, hid2weight_jac, hid2hid_jac)

        def scan_over_multi_times(
            args_at_multi_times: Dict,  # the inputs for multiple time steps
            hidden_vals_,
            other_vals_
        ):
            # KEY: always return output at multi-time steps
            if len(args_at_multi_times):
                return jax.lax.scan(scan_fn, (hidden_vals_, other_vals_), args_at_multi_times)
            else:
                r = scan_fn((hidden_vals_, other_vals_), args_at_multi_times)
                return r[0], jax.tree.map(lambda x: jnp.expand_dims(x, 0), r[1])

        # processing the inputs information
        args_single_step, args_multi_steps, tree_def = split_data_types(*args)

        # check the batch size
        if len(args_multi_steps):
            args_dim = [jnp.shape(x)[0] for x in jax.tree.leaves(args_multi_steps)]
            if len(set(args_dim)) != 1:
                raise ValueError(f'The sequence size should be the same for all inputs. But we got {args_dim}.')

        (
            (
                hidden_vals,
                oth_state_vals
            ),
            (
                outs_multi_steps,
                hid2weight_jac_multi_steps,
                hid2hid_jac_multi_steps
            )
        ) = scan_over_multi_times(
            args_multi_steps,
            hidden_vals,
            other_vals
        )

        # recovering the other non-etrace weights, although the weights are not changed
        assign_dict_state_values(non_etrace_weight_states, non_etrace_weight_vals, write=False)
        assign_dict_state_values(etrace_param_states, etrace_weight_vals, write=False)

        return (
            outs_multi_steps,
            hidden_vals,
            oth_state_vals,
            hid2weight_jac_multi_steps,
            hid2hid_jac_multi_steps
        )

    def solve_h2w_h2h_jacobian_and_l2h_vjp(
        self,
        *args,
    ) -> Tuple[Outputs, HiddenVals, StateVals, Hid2WeightJacobian, Hid2HidJacobian, Residuals]:
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

        # ---------------------- [Part 1] ----------------------
        # weights, hidden, and states information
        # for VJP computation
        # ------------------------------------------------------

        #  [KEY]
        #  The most important assumption here is
        #  that the weight values (including etrace weights and normal param weights) are not changed

        # split the states, got initial hidden and weight values

        (
            etrace_param_states,
            hidden_states,
            non_etrace_weight_states,
            other_states
        ) = split_dict_states_v2(self.states)
        if self.vjp_method != 'multi-step':
            etrace_weight_vals = dict()
            hidden_perturbs = [
                u.math.zeros(v.aval.shape, v.aval.dtype)
                for v in self.compiled.out_hidden_jaxvars
            ]
            etrace_weight_vals_restore = {path: st.value for path, st in etrace_param_states.items()}
        else:
            etrace_weight_vals = {path: st.value for path, st in etrace_param_states.items()}
            etrace_weight_vals_restore = etrace_weight_vals
            hidden_perturbs = []
        non_etrace_weight_vals = {path: st.value for path, st in non_etrace_weight_states.items()}
        hidden_vals = {path: st.value for path, st in hidden_states.items()}
        other_vals = {path: st.value for path, st in other_states.items()}

        def fun_for_vjp(
            inputs,  # functional inputs, original inputs
            hiddens,  # etrace hidden states
            non_etrace_weights,  # non-etrace weights
            etrace_weights,  # etrace weights
            oth_states,  # other states
            perturbs  # hidden perturbations, useful when computing \partial L / \partial h
        ):
            # assign state values
            if len(etrace_weights) > 0:
                assign_dict_state_values(etrace_param_states, etrace_weights, write=False)
            assign_dict_state_values(hidden_states, hiddens, write=False)
            assign_dict_state_values(non_etrace_weight_states, non_etrace_weights, write=False)
            assign_dict_state_values(other_states, oth_states, write=False)

            # get state values by the "stateful_model", to preserve the order of states
            old_state_vals = [st.value for st in self.compiled.stateful_fn_states]

            if self.is_single_step:
                assert self.compiled.jaxpr_perturb_hidden is not None, (
                    'The jaxpr_with_hidden_perturb should not be None '
                    'when the vjp_time_ahead is 0.'
                )
                jaxpr_outs = jax.core.eval_jaxpr(
                    self.compiled.jaxpr_perturb_hidden.jaxpr,
                    self.compiled.jaxpr_perturb_hidden.consts,
                    *jax.tree.leaves((inputs, old_state_vals, perturbs))
                )

            else:
                assert self.compiled.augmented_jaxpr is not None, (
                    'The augmented_jaxpr should not be None '
                    'when the vjp_time_ahead > 0.'
                )
                # calling the function
                jaxpr_outs = jax.core.eval_jaxpr(
                    self.compiled.augmented_jaxpr.jaxpr,
                    self.compiled.augmented_jaxpr.consts,
                    *jax.tree.leaves((inputs, old_state_vals))
                )

            # --- intermediate values --- #
            temps = {
                v: r
                for v, r in zip(
                    self.compiled.out_all_jaxvars[self.compiled.num_out:],
                    jaxpr_outs[self.compiled.num_out:]
                )
            }

            # --- outputs  --- #
            state_outs = [temps[v] for v in self.compiled.out_state_jaxvars]
            out, new_state_vals = self.compiled.stateful_fn_outtree.unflatten(
                jaxpr_outs[:self.compiled.num_out] + state_outs
            )

            # --- compute the hidden-to-weight Jacobian --- #
            hid2weight_jac = self._compute_hid2weight_jacobian(temps)

            # --- compute the hidden-to-hidden Jacobian --- #
            hid2hid_jac = self._compute_hid2hid_jacobian(temps)

            # ---- compute new state values ---- #
            # get new state values, do not return the weight values, since they are not changed
            new_state_vals = {
                self.state_id_to_path[id(st)]: st_val
                for st, st_val in zip(self.compiled.stateful_fn_states, new_state_vals)
            }
            (
                _,  # drop weights, since they are not changed
                new_hiddens,
                new_others
            ) = dict_split_state_values(self.states, new_state_vals)
            return out, new_hiddens, new_others, hid2weight_jac, hid2hid_jac

        # ---------------------- [Part 2] ----------------------
        # Scan VJP function over multiple time steps
        # ------------------------------------------------------

        def scan_over_multiple_inputs(
            inputs: Dict,  # the inputs for multiple time steps
            hidden_vals_,  # the initial hidden states
            non_etrace_weight_vals_,  # the non-etrace weights
            etrace_weight_vals_,  # the etrace weights
            other_vals_,  # the initial other states
            hidden_perturbs_  # the hidden perturbations, only used when vjp_time_ahead == 0
        ):

            # processing the inputs information
            args_single_step, args_multi_steps, tree_def = split_data_types(*inputs)

            if len(args_multi_steps):
                # check the batch size
                args_dim = [jnp.shape(x)[0] for x in jax.tree.leaves(args_multi_steps)]
                if len(set(args_dim)) != 1:
                    raise ValueError(f'The sequence size should be the same for all inputs. But we got {args_dim}.')

            def scan_fn(carray, x_single_step: Dict):
                args_ = merge_data(tree_def, x_single_step, args_single_step)

                hidden_vals_, other_vals_ = carray
                (
                    out,
                    new_hiddens,
                    new_others,
                    hid2weight_jac,
                    hid2hid_jac
                ) = fun_for_vjp(
                    args_,
                    hidden_vals_,
                    non_etrace_weight_vals_,
                    etrace_weight_vals_,
                    other_vals_,
                    hidden_perturbs_
                )

                return (
                    (new_hiddens, new_others),
                    (out, hid2weight_jac, hid2hid_jac)
                )

            if len(args_multi_steps):
                (
                    (
                        new_hiddens,
                        new_others
                    ),
                    (
                        _outs_multi_steps,
                        _hid2weight_jac_multi_steps,
                        _hid2hid_jac_multi_steps
                    )
                ) = jax.lax.scan(scan_fn, (hidden_vals_, other_vals_), args_multi_steps)

            else:
                (
                    (
                        new_hiddens,
                        new_others
                    ),
                    ret
                ) = scan_fn((hidden_vals_, other_vals_), args_multi_steps)
                (
                    _outs_multi_steps,
                    _hid2weight_jac_multi_steps,
                    _hid2hid_jac_multi_steps
                ) = jax.tree.map(lambda x: jnp.expand_dims(x, 0), ret)

            return (
                (
                    _outs_multi_steps,
                    new_hiddens,
                    new_others
                ),
                (
                    _hid2weight_jac_multi_steps,
                    _hid2hid_jac_multi_steps
                )
            )

        # ---------------------- [Part 3] ------------------------
        # Compile the AutoGrad of the VJP function that over time
        # into the residual jaxpr representation
        # ---------------------------------------------------------

        # format VJP calling, compile the autograd information into the residual jaxpr representation
        # so that it can be computed when they are needed.
        (
            (
                outs_multi_steps,
                hidden_vals,
                other_vals
            ),
            f_vjp,
            (
                hid2weight_jac_multi_steps,
                hid2hid_jac_multi_steps
            )
        ) = jax.vjp(
            scan_over_multiple_inputs,  # the function
            args,  # the inputs (multiple time)
            hidden_vals,  # the inputs (single time)
            non_etrace_weight_vals,  # the inputs (single time)
            etrace_weight_vals,  # the inputs (single time)
            other_vals,  # the inputs (single time)
            hidden_perturbs,  # the inputs (single time)
            has_aux=True
        )
        out_flat, out_tree = jax.tree.flatten(((outs_multi_steps, hidden_vals, other_vals),))
        rule, in_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(f_vjp), out_tree)
        out_avals = [jax.core.get_aval(x).at_least_vspace() for x in out_flat]
        jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(rule, out_avals)
        residual = Residuals(jaxpr, in_tree(), out_tree, consts)

        # ---------------------- [Part 4] ------------------------
        # Recover the weight states values
        # ---------------------------------------------------------

        # recovering the other non-etrace weights, although the weights are not changed
        assign_dict_state_values(non_etrace_weight_states, non_etrace_weight_vals, write=False)
        assign_dict_state_values(etrace_param_states, etrace_weight_vals_restore, write=False)

        return (
            outs_multi_steps,
            hidden_vals,
            other_vals,
            hid2weight_jac_multi_steps,
            hid2hid_jac_multi_steps,
            residual
        )
