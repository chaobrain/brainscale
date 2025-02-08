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
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
#
# Refinement History:
#    [2025-02-06]
#       - [x] split into "_etrace_algorithms.py" and "_etrace_vjp_algorithms.py"
#
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Dict, Tuple, Any, List, Optional, Sequence

import brainstate as bst
import brainunit as u
import jax.core
import jax.numpy as jnp

from ._etrace_algorithms import (
    ETraceAlgorithm,
    EligibilityTraceData,
)
from ._etrace_compiler_graph import (
    HiddenParamOpRelation,
    HiddenGroup,
)
from ._etrace_concepts import (
    ETraceParam,
    ElemWiseParam,
    ETraceGrad,
)
from ._etrace_input_data import has_multistep_data
from ._etrace_operators import ETraceOp
from ._etrace_vjp_compiler import CompiledVjpGraph
from ._etrace_vjp_graph import ETraceVjpGraphExecutor
from ._misc import (
    check_dict_keys,
    hid_group_key,
    etrace_param_key,
    etrace_df_key,
)
from ._state_managment import (
    assign_state_values_v2
)
from ._typing import (
    PyTree,
    Outputs,
    WeightID,
    WeightVals,
    HiddenVals,
    StateVals,
    ETraceVals,
    Path,
    ETraceX_Key,
    ETraceDF_Key,
    ETraceWG_Key,
    Hid2WeightJacobian,
    Hid2HidJacobian,
    HidGroupJacobian,
    dG_Inputs,
    dG_Weight,
    dG_Hidden,
    dG_State,
)

__all__ = [
    'ETraceVjpAlgorithm',  # the base class for the eligibility trace algorithm with the VJP gradient computation
    'IODimVjpAlgorithm',  # the diagonally approximated algorithm with the input-output dimension complexity
    'ParamDimVjpAlgorithm',  # the diagonally approximated algorithm with the parameter dimension complexity
    'HybridDimVjpAlgorithm',  # the diagonally approximated algorithm with hybrid complexity (either I/O or parameter)
]


def _format_decay_and_rank(decay_or_rank) -> Tuple[float, int]:
    """
    Format the decay or the rank of the approximation.

    Args:
      decay_or_rank: float, int, the decay factor or the number of approximation rank.

    Returns:
      Tuple[float, int], the decay factor and the number of approximation rank.
    """
    # number of approximation rank and the decay factor
    if isinstance(decay_or_rank, float):
        assert 0 < decay_or_rank < 1, f'The decay should be in (0, 1). While we got {decay_or_rank}. '
        decay = decay_or_rank  # (num_rank - 1) / (num_rank + 1)
        num_rank = round(2. / (1 - decay) - 1)
    elif isinstance(decay_or_rank, int):
        assert decay_or_rank > 0, f'The num_rank should be greater than 0. While we got {decay_or_rank}. '
        num_rank = decay_or_rank
        decay = (num_rank - 1) / (num_rank + 1)  # (num_rank - 1) / (num_rank + 1)
    else:
        raise ValueError('Please provide "num_rank" (int) or "decay" (float, 0 < decay < 1). ')
    return decay, num_rank


def _expon_smooth(old, new, decay):
    """
    Exponential smoothing.

    :param old: the old value
    :param new: the new value
    :param decay: the decay factor
    :return: the smoothed value
    """
    if new is None:
        return decay * old
    return decay * old + (1 - decay) * new


def _low_pass_filter(old, new, alpha):
    """
    Low-pass filter.

    :param old: the old value
    :param new: the new value
    :param alpha: the filter factor
    :return: the filtered value
    """
    if new is None:
        return alpha * old
    return alpha * old + new


def _update_dict(
    the_dict: Dict,
    key: Any,
    value: PyTree,
    error_when_no_key: Optional[bool] = False
):
    """Update the dictionary.

    If the key exists, then add the value to the existing value.
    Otherwise, create a new key-value pair.

    Args:
      the_dict: The dictionary.
      key: The key.
      value: The value.
      error_when_no_key: bool, whether to raise an error when the key does not exist.

    """
    old_value = the_dict.get(key, None)
    if old_value is None:
        if error_when_no_key:
            raise ValueError(f'The key {key} does not exist in the dictionary. ')
        the_dict[key] = value
    else:
        the_dict[key] = jax.tree.map(u.math.add, old_value, value, is_leaf=lambda x: isinstance(x, u.Quantity))


def _batched_zeros_like(batch_size: Optional[int],
                        x: jax.Array):
    """
    Create a batched zeros array like the input array.

    Args:
      batch_size: int, the batch size.
      x: jax.Array, the input array.

    Returns:
      jax.Array, the batched zeros array.
    """
    if batch_size is None:
        return u.math.zeros_like(x)
    else:
        return u.math.zeros((batch_size,) + x.shape, x.dtype)


def _init_IO_dim_state(
    etrace_xs: Dict[ETraceX_Key, bst.State],
    etrace_dfs: Dict[ETraceDF_Key, bst.State],
    etrace_xs_to_weights: defaultdict[ETraceX_Key, List[Path]],
    state_id_to_path: Dict[int, Path],
    relation: HiddenParamOpRelation,
):
    # For the relation
    #
    #   h1, h2, ... = f(x, w)
    #
    # we need to initialize the eligibility trace states for the weight x and the df.

    # "relation.x" may be repeatedly used in the graph
    if not isinstance(relation.weight, ElemWiseParam):
        if relation.x not in etrace_xs:
            shape = relation.x.aval.shape
            dtype = relation.x.aval.dtype
            etrace_xs[relation.x] = EligibilityTraceData(u.math.zeros(shape, dtype))

        # relation.x maybe repeatedly used to feed into the
        # weight operation for transforming the hidden states
        # therefore we record the target paths of the weight x
        #
        etrace_xs_to_weights[relation.x].append(state_id_to_path[id(relation.weight)])

    y_shape = relation.y.aval.shape
    y_dtype = relation.y.aval.dtype
    for group in relation.hidden_groups:
        group: HiddenGroup
        assert y_shape == group.varshape, (
            f'The shape of the hidden states should be the same as the shape of the hidden group. '
            f'While we got {y_shape} != {group.varshape}. '
        )
        key = etrace_df_key(relation.y, group.index)
        if key in etrace_dfs:  # relation.y is a unique output of the weight operation
            raise ValueError(f'The relation {key} has been added. ')

        #
        # Group 1:
        #
        #   [∂a^t-1/∂θ1, ∂b^t-1/∂θ1, ...]
        #
        # Group 2:
        #
        #   [∂A^t-1/∂θ1, ∂B^t-1/∂θ1, ...]
        #
        shape = y_shape + (group.num_state,)
        etrace_dfs[key] = EligibilityTraceData(u.math.zeros(shape, y_dtype))


def _update_IO_dim_etrace_scan_fn(
    hist_etrace_vals: Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    hid_weight_op_relations: Sequence[HiddenParamOpRelation],
    decay: float,
):
    # --- the data --- #

    #
    # the etrace data at the current time step (t) of the O(n) algorithm
    # is a tuple, including the weight x and df values.
    #
    # For the weight x, it is a dictionary,
    #    {ETraceX_Key: jax.Array}
    #
    # For the weight df, it is a dictionary,
    #    {ETraceDF_Key: jax.Array}
    #
    xs: Dict[ETraceX_Key, jax.Array] = jacobians[0]
    dfs: Dict[ETraceDF_Key, jax.Array] = jacobians[1]

    #
    # the hidden-to-hidden Jacobians
    #
    hid_group_jacobians: Sequence[jax.Array] = jacobians[2]

    #
    # the history etrace values
    #
    # - hist_xs is a dictionary,
    #       {ETraceX_Key: brainstate.State}
    #
    # - hist_dfs is a dictionary,
    #       {ETraceDF_Key: brainstate.State}
    #
    hist_xs, hist_dfs = hist_etrace_vals

    #
    # the new etrace values
    #
    new_etrace_xs, new_etrace_dfs = dict(), dict()

    # --- the update --- #

    #
    # Step 1:
    #
    #   update the weight x using the equation:
    #           x^t = α * x^t-1 + x^t, where α is the decay factor.
    #
    check_dict_keys(hist_xs, xs)
    for xkey in hist_xs.keys():
        new_etrace_xs[xkey] = _low_pass_filter(hist_xs[xkey], xs[xkey], decay)

    for relation in hid_weight_op_relations:
        relation: HiddenParamOpRelation

        for group in relation.hidden_groups:
            group: HiddenGroup

            #
            # Step 2:
            #
            # update the eligibility trace * hidden diagonal Jacobian
            #         dϵ^t_{pre} = D_h ⊙ dϵ^t-1, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            #
            # JVP equation for the following Jacobian computation:
            #
            # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
            #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
            #
            # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ2,
            #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
            #
            df_key = etrace_df_key(relation.y, group.index)
            hid_jac = hid_group_jacobians[group.index]
            pre_trace_df = jnp.einsum(
                '...ij,...j->...i',
                hid_jac,
                hist_dfs[df_key]
            )

            #
            # Step 3:
            #
            # update: eligibility trace * hidden diagonal Jacobian + new hidden df
            #        dϵ^t = dϵ^t_{pre} + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            new_etrace_dfs[df_key] = _expon_smooth(pre_trace_df, dfs[df_key], decay)

    return (new_etrace_xs, new_etrace_dfs), None


def _solve_IO_dim_weight_gradients(
    hist_etrace_data: Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ],
    dG_weights: Dict[Path, dG_Weight],
    dG_hidden_groups: Sequence[jax.Array],  # same length as total hidden groups
    weight_hidden_relations: Sequence[HiddenParamOpRelation],
    running_index: int,
    decay: float,
):
    #
    # Avoid the exponential smoothing bias at the beginning.
    # This is the correction factor for the exponential smoothing.
    #
    correction_factor = 1. - u.math.power(1. - decay, running_index + 1)
    correction_factor = u.math.where(running_index < 1000, correction_factor, 1.)
    correction_factor = jax.lax.stop_gradient(correction_factor)

    xs, dfs = hist_etrace_data

    for relation in weight_hidden_relations:
        relation: HiddenParamOpRelation

        if not isinstance(relation.weight, ElemWiseParam):
            x = xs[relation.x]
        else:
            x = None
        weight_path = relation.path
        weight_op = relation.weight.op

        for group in relation.hidden_groups:
            group: HiddenGroup
            #
            # Step 4:
            #
            # Solve the weight gradients by using the etrace data
            #
            #       dw = (dL/dH \circ df) \otimes x
            #
            df_key = etrace_df_key(relation.y, group.index)
            df = dfs[df_key] / correction_factor  # the hidden gradients
            df_hid = df * dG_hidden_groups[group.index]  # the hidden gradients

            #
            # Compute the weight gradients according to the x and y
            #
            #    dw = df(dx, dy)
            #
            dg_weight = weight_op.xy_to_w(x, df_hid)

            # update the weight gradients
            _update_dict(dG_weights, weight_path, dg_weight)  # update the weight gradients


def _init_param_dim_state(
    mode: bst.mixin.Mode,
    etrace_bwg: Dict[ETraceWG_Key, bst.State],
    relation: HiddenParamOpRelation
):
    # For the relation
    #
    #   h1, h2, ... = f(x, w)
    #
    # we need to initialize the eligibility trace states for the weight x and the df.

    # TODO: assume the batch size is the first dimension
    y_shape = relation.y.aval.shape
    batch_size = y_shape[0] if mode.has(bst.mixin.Batching) else None
    for group in relation.hidden_groups:
        group: HiddenGroup
        bwg_key = etrace_param_key(relation.path, relation.y, group.index)
        if bwg_key in etrace_bwg:  # The key should be unique
            raise ValueError(f'The relation {bwg_key} has been added. ')
        etrace_bwg[bwg_key] = EligibilityTraceData(
            jax.tree.map(partial(_batched_zeros_like, batch_size),
                         relation.weight.value)
        )


def _update_param_dim_etrace_scan_fn(
    hist_etrace_vals: Dict[ETraceWG_Key, jax.Array],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    weight_path_to_vals: Dict[Path, PyTree],
    hidden_param_op_relations,
    mode: bst.mixin.Mode,
):
    # --- the data --- #

    #
    # + "hist_etrace_vals" has the following structure:
    #    - key: the weight id, the weight-x jax var, the hidden state var
    #    - value: the batched weight gradients
    #

    # + "hid2weight_jac" has the following structure:
    #    - a dict of weight x gradients
    #       * key: the weight x jax var
    #       * value: the weight x gradients
    #    - a dict of weight y gradients
    #       * key: the tuple of the weight y jax var and the hidden state jax var
    #       * value: the weight y gradients
    #
    etrace_xs_at_t: Dict[ETraceX_Key, jax.Array] = jacobians[0]
    etrace_ys_at_t: Dict[ETraceDF_Key, jax.Array] = jacobians[1]

    #
    # the hidden-to-hidden Jacobians
    #
    hid_group_jacobians: Sequence[jax.Array] = jacobians[2]

    #
    # The etrace weight gradients at the current time step.
    # i.e., The "hist_etrace_vals" at the next time step
    #
    new_etrace_bwg = dict()

    for relation in hidden_param_op_relations:
        relation: HiddenParamOpRelation

        #
        # Step 1:
        #
        # Necessary information for the etrace computation
        #
        # 1. the etrace operation for computing etrace updates
        # 2. the weight information
        # 3. the operator information
        #
        weight_path = relation.path
        weight_val = weight_path_to_vals[weight_path]
        etrace_op: ETraceOp = relation.weight.op
        if isinstance(etrace_op, ElemWiseParam):
            x = None
            fn_dw = lambda df: jax.vjp(etrace_op.xw_to_y, weight_val)[1]((df,))
        else:
            x = etrace_xs_at_t[relation.x]
            fn_dw = lambda x, df: jax.vjp(etrace_op.xw_to_y, x, weight_val)[1]((df,))
        if mode.has(bst.mixin.Batching):
            fn_dw = jax.vmap(fn_dw)

        for group in relation.hidden_groups:
            group: HiddenGroup

            #
            # Step 2:
            #
            # compute the current step weight gradients:
            #
            #       \partial h^t / \partial W^t = vjp(f(x, w))(df)
            #
            df = etrace_ys_at_t[(relation.y, hid_group_key(group.index))]
            #
            # vmap over the different hidden states,
            #
            # x: (n_input, ..., )
            # df: (n_hidden, ..., n_state)
            # phg_to_pw: (n_param, ..., n_state)
            if isinstance(etrace_op, ElemWiseParam):
                phg_to_pw = jax.vmap(fn_dw, in_axes=-1, out_axes=-1)(df)
            else:
                phg_to_pw = jax.vmap(fn_dw, in_axes=(None, -1), out_axes=-1)(x, df)

            #
            # Step 3:
            #
            # computing the following vector-Jacobian product:
            #  ϵ^t_{pre} = D_h ⊙ ϵ^{t-1}
            #
            # i.e., the hidden-to-hidden Jacobian diagonal matrix * the hidden df at the previous time step
            #
            #  ∂V^t/∂θ1 = ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ...
            #
            w_key = etrace_param_key(weight_path, relation.y, group.index)
            old_bwg = hist_etrace_vals[w_key]
            diag = hid_group_jacobians[group.index]
            #
            # vmap over j, over the different hidden states \partial h_i^t / \partial h_j^t
            #
            # diag: (n_hidden, ..., n_state, [n_state])
            # old_bwg: (n_param, ..., [n_state])
            fn_bwg_pre = lambda d: jax.vmap(etrace_op.yw_to_w, in_axes=-1, out_axes=-1)(d, old_bwg).sum(axis=-1)
            #
            # vmap over i, over the different hidden states \partial h_i^t / \partial h_j^t
            #
            # diag: (n_hidden, ..., [n_state], n_state)
            # old_bwg: (n_param, ..., n_state)
            # new_bwg_pre: (n_param, ..., n_state)
            new_bwg_pre = jax.vmap(fn_bwg_pre, in_axes=-2)(diag)

            #
            # Step 4:
            #
            # update: eligibility trace * hidden diagonal Jacobian + new hidden df
            #        ϵ^t = ϵ^t_{pre} + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            new_etrace_bwg[w_key] = jax.tree.map(u.math.add, new_bwg_pre, phg_to_pw, is_leaf=u.math.is_quantity)

    return new_etrace_bwg, None


def _solve_param_dim_weight_gradients(
    hist_etrace_data: Dict[ETraceWG_Key, PyTree],  # the history etrace data
    dG_weights: Dict[Path, dG_Weight],  # weight gradients
    dG_hidden_groups: Sequence[jax.Array],  # hidden group gradients
    weight_hidden_relations: Sequence[HiddenParamOpRelation],
    mode: bst.mixin.Mode,
):
    # update the etrace weight gradients
    temp_data = dict()
    for relation in weight_hidden_relations:

        #
        # Step 1:
        #
        # Necessary information for the etrace computation
        #
        # 1. the etrace operation for computing etrace updates
        # 2. the weight information
        # 3. the operator information
        #
        weight_path = relation.path
        etrace_op: ETraceOp = relation.weight.op
        yw_to_w = jax.vmap(etrace_op.yw_to_w) if mode.has(bst.mixin.Batching) else etrace_op.yw_to_w

        for group in relation.hidden_groups:
            group: HiddenGroup

            #
            # Step 2:
            #
            # compute the weight gradients:
            #
            #   dE/dW = dE/dH * dH/dW, computing the final weight gradients
            #
            w_key = etrace_param_key(weight_path, relation.y, group.index)
            etrace_data = hist_etrace_data[w_key]
            dg_hidden = dG_hidden_groups[group.index]
            #
            # etrace_data: [n_batch, n_param, ..., n_state]
            #               or,
            #              [n_param, ..., n_state]
            # dg_hidden:   [n_batch, n_hidden, ..., n_state]
            #               or,
            #              [n_hidden, ..., n_state]
            dg_weight = jax.vmap(yw_to_w, in_axes=-1, out_axes=-1)(dg_hidden, etrace_data).sum(axis=-1)

            # update the weight gradients
            _update_dict(temp_data, weight_path, dg_weight)

    #
    # Step 3:
    #
    # sum up the batched weight gradients
    if mode.has(bst.mixin.Batching):
        for key, val in temp_data.items():
            temp_data[key] = jax.tree_map(lambda x: u.math.sum(x, axis=0), val)

    # update the weight gradients
    for key, val in temp_data.items():
        _update_dict(dG_weights, key, val)


def _zeros_like_batch_or_not(
    batch_size: Optional[int],
    x: jax.Array
):
    """
    Create a batched zeros array like the input array.

    Args:
      batch_size: int, the batch size.
      x: jax.Array, the input array.

    Returns:
      jax.Array, the batched zeros array.
    """
    if batch_size is not None:
        assert isinstance(batch_size, int), 'The batch size should be an integer. '
        return u.math.zeros((batch_size,) + x.shape[1:], x.dtype)
    else:
        return u.math.zeros_like(x)


def _reset_state_in_a_dict(
    state_dict: Dict[Any, bst.State],
    batch_size: Optional[int],
):
    for k, v in state_dict.items():
        state_dict[k].value = jax.tree.map(partial(_zeros_like_batch_or_not, batch_size), v)


def _numel(pytree: PyTree):
    return sum(u.math.size(x) for x in jax.tree_leaves(pytree))


def _is_weight_need_full_grad(
    relation: HiddenParamOpRelation,
    mode: bst.mixin.Mode
):
    if isinstance(relation.weight, ETraceParam):
        #
        # When
        #     weight.gradient == ETraceGrad.full
        #
        # the weights will be forced to use O(n^2) algorithm
        # to compute the eligibility trace.
        #
        if relation.weight.gradient == ETraceGrad.full:
            return True

        #
        # When
        #     weight.gradient == ETraceGrad.approx
        #
        # the weights will be forced to use O(n) algorithm
        # to compute the eligibility trace.
        #
        if relation.weight.gradient == ETraceGrad.approx:
            return False

    if isinstance(relation.weight, ElemWiseParam):
        #
        # When
        #     weight is an element-wise parameter
        #
        # the weights will be forced to use O(n^2) algorithm
        # to compute the eligibility trace.
        #
        return True

    batch_size = relation.x.aval.shape[0] if mode.has(bst.mixin.Batching) else 1
    if _numel(relation.x) + _numel(relation.y) > batch_size * _numel(relation.weight.value):
        #
        # When the number of elements in the inputs and outputs are bigger than the weight number,
        # we will use the O(n^2) algorithm to compute the eligibility trace, since
        # storing the batched weight gradients will be less expensive.
        #
        return True

    else:
        #
        # For most cases, we will use the O(n) algorithm to compute the eligibility trace.
        # Since the number of elements in input and output (I + O) is greatly less than the number
        # of elements in the weight (W = I * O).
        #
        return False


class ETraceVjpAlgorithm(ETraceAlgorithm):
    r"""
    The base class for the eligibility trace algorithm which supporting the VJP gradient
    computation (reverse-mode differentiation).

    The term ``VJP`` comes from the following two aspects:

    **First**, this module is designed to be compatible with the JAX's VJP mechanism.
    This means that the gradient is computed according to the reverse-mode differentiation
    interface, like the ``jax.grad()`` function, the ``jax.vjp()`` function,
    or the ``jax.jacrev()`` function. The true update function is defined as a custom
    VJP function ``._true_update_fun()``, which receives the inputs, the hidden states,
    other states, and etrace variables at the last time step, and returns the outputs,
    the hidden states, other states, and etrace variables at the current time step.

    For each subclass (or the instance of an etrace algorithm), we should define the
    following methods:

    - ``._update()``: update the eligibility trace states and return the outputs, hidden states, other states, and etrace data.
    - ``._update_fwd()``: the forward pass of the custom VJP rule.
    - ``._update_bwd()``: the backward pass of the custom VJP rule.

    However, this class has provided a default implementation for the ``._update()``,
    ``._update_fwd()``, and ``._update_bwd()`` methods.

    To implement a new etrace algorithm, users just need to override the following methods:

    - ``._solve_weight_gradients()``: solve the gradients of the learnable weights / parameters.
    - ``._update_etrace_data()``: update the eligibility trace data.
    - ``._assign_etrace_data()``: assign the eligibility trace data to the states.
    - ``._get_etrace_data()``: get the eligibility trace data.

    **Second**, the algorithm computes the spatial gradient $\partial L^t / \partial H^t$ using the standard
    back-propagation algorithm. This design can enhance the accuracy and the stability of the algorithm for
    computing gradients.


    Parameters
    ----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    name: str, optional
        The name of the etrace algorithm.
    vjp_method: str
        The time to compute the loss-to-hidden Jacobian.

        - ``0``: the current time step: $\frac{\partial L^t}{\partial h^t}$.  Memory is
        - ``1``: the last time step: $\frac{\partial L^{t-1}}{\partial h^{t-1}}$.
        - ``k``: the t-k time step: $\frac{\partial L^{t-k}}{\partial h^{t-k}}$.

    """

    __module__ = 'brainscale'
    graph: ETraceVjpGraphExecutor
    compiled: CompiledVjpGraph

    def __init__(
        self,
        model: bst.nn.Module,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):

        # the VJP method
        assert vjp_method in ('single-step', 'multi-step'), (
            'The VJP method should be either "single-step" or "multi-step". '
            f'While we got {vjp_method}. '
        )
        self.vjp_method = vjp_method

        # graph
        graph = ETraceVjpGraphExecutor(model, vjp_method=vjp_method)

        # super initialization
        super().__init__(model=model, name=name, graph=graph)

        # the update rule
        self._true_update_fun = jax.custom_vjp(self._update_fn)
        self._true_update_fun.defvjp(
            fwd=self._update_fn_fwd,
            bwd=self._update_fn_bwd
        )

    def _assert_compiled(self):
        if not self.is_compiled:
            raise ValueError('The etrace algorithm has not been compiled. Please call `compile_graph()` first. ')

    def update(self, *args) -> Any:
        """
        Update the model states and the eligibility trace.

        The input arguments ``args`` here supports very complex data structures, including
        the combination of :py:class:`SingleStepData` and :py:class:`MultiStepData`.

        - :py:class:`SingleStepData`: indicating the data at the single time step, $x_t$.
        - :py:class:`MultiStepData`: indicating the data at multiple time steps, $[x_{t-k}, ..., x_t]$.

        Suppose all inputs have the shape of ``(10,)``.

        If the input arguments are given by:

        .. code-block:: python

            x = [jnp.ones((10,)), jnp.zeros((10,))]

        Then, two input arguments are considered as the :py:class:`SingleStepData`.

        If the input arguments are given by:

        .. code-block:: python

            x = [brainscale.SingleStepData(jnp.ones((10,))),
                 brainscale.SingleStepData(jnp.zeros((10,)))]

        This is the same as the previous case, they are all considered as the input at the current time step.

        If the input arguments are given by:

        .. code-block:: python

            x = [brainscale.MultiStepData(jnp.ones((5, 10)),
                 jnp.zeros((10,)))]

        or,

        .. code-block:: python

            x = [brainscale.MultiStepData(jnp.ones((5, 10)),
                 brainscale.SingleStepData(jnp.zeros((10,)))]

        Then, the first input argument is considered as the :py:class:`MultiStepData`, and its data will
        be fed into the model within five consecutive steps, and the second input argument will be fed
        into the model at each time of this five consecutive steps.

        Args:
            *args: the input arguments.
        """

        # ----------------------------------------------------------------------------------------------
        #
        # This method is the main function to
        #
        # - update the model
        # - update the eligibility trace states
        # - compute the weight gradients
        #
        # The key here is that we change the object-oriented attributes as the function arguments.
        # Therefore, the function arguments are the states of the current time step, and the function
        # returns the states of the next time step.
        #
        # Particularly, the model calls the "_true_update_fun()" function to update the states.
        #
        # ----------------------------------------------------------------------------------------------

        #
        # This function need to process the following multiple cases:
        #
        # 1. if vjp_method = 'single-step', input = SingleStepData, then output is single step
        #
        # 2. if vjp_method = 'single-step', input = MultiStepData, then output is multiple step data
        #
        # 3. if vjp_method = 'multi-step', input = SingleStepData, then output is single step
        #
        # 4. if vjp_method = 'multi-step', input = MultiStepData, then output is multiple step data
        #

        input_is_multi_step = has_multistep_data(*args)

        # check the compilation
        self._assert_compiled()

        # state values
        weight_vals = {
            key: st.value
            for key, st in self.param_states.items()
        }
        hidden_vals = {
            key: st.value
            for key, st in self.hidden_states.items()
        }
        other_vals = {
            key: st.value
            for key, st in self.other_states.items()
        }
        # etrace data
        last_etrace_vals = self._get_etrace_data()

        # update all states
        #
        # [KEY] The key here is that we change the object-oriented attributes as the function arguments.
        #       Therefore, the function arguments are the states of the current time step, and the function
        #       returns the states of the next time step.
        #
        # out: is always multiple step
        (
            out,
            hidden_vals,
            other_vals,
            new_etrace_vals
        ) = self._true_update_fun(
            args,
            weight_vals,
            hidden_vals,
            other_vals,
            last_etrace_vals,
            self.running_index.value
        )

        # assign/restore the weight values back
        #
        # [KEY] assuming the weight values are not changed
        #       This is a key assumption in the RTRL algorithm.
        #       This is very important for the implementation.
        assign_state_values_v2(self.param_states, weight_vals, write=False)

        # assign the new hidden and state values
        assign_state_values_v2(self.hidden_states, hidden_vals)
        assign_state_values_v2(self.other_states, other_vals)

        #
        # assign the new etrace values
        #
        # "self._assign_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        self._assign_etrace_data(new_etrace_vals)  # call the protocol method

        # update the running index
        running_index = self.running_index.value + 1
        self.running_index.value = jax.lax.stop_gradient(jnp.where(running_index >= 0, running_index, 0))

        # return the model output
        return (
            out
            if input_is_multi_step else
            jax.tree.map(lambda x: x[0], out)
        )

    def _update_fn(
        self,
        args,
        weight_vals: WeightVals,
        hidden_vals: HiddenVals,
        oth_state_vals: StateVals,
        etrace_vals: ETraceVals,
        running_index,
    ) -> Tuple[Outputs, HiddenVals, StateVals, ETraceVals]:
        """
        The main function to update the [model] and the [eligibility trace] states.

        Particularly, ``self.graph.solve_h2w_h2h_jacobian()`` is called to:
          - compute the model output, the hidden states, and the other states
          - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian

        Then, ``self._update_etrace_data`` is called to:
          - update the eligibility trace data

        Moreover, this function returns:
          - the model output
          - the updated hidden states
          - the updated other states
          - the updated eligibility trace states

        Note that the weight values are assumed not changed in this function.

        """

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, oth_state_vals, write=False)

        # necessary jacobian information of the weights
        (
            out,
            hidden_vals,
            oth_state_vals,
            hid2weight_jac_multi_steps,
            hid2hid_jac_multi_steps
        ) = self.graph.solve_h2w_h2h_jacobian(*args)

        # eligibility trace update
        #
        # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        etrace_vals = self._update_etrace_data(
            running_index,
            etrace_vals,
            hid2weight_jac_multi_steps,
            hid2hid_jac_multi_steps,
            weight_vals,
        )

        # returns
        return out, hidden_vals, oth_state_vals, etrace_vals

    def _update_fn_fwd(
        self,
        args,
        weight_vals: WeightVals,
        hidden_vals: HiddenVals,
        othstate_vals: StateVals,
        etrace_vals: ETraceVals,
        running_index: int,
    ) -> Tuple[Tuple[Outputs, HiddenVals, StateVals, ETraceVals], Any]:
        """
        The forward function to update the [model] and the [eligibility trace] states when computing
        the VJP gradients.

        Particularly, ``self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp()`` is called to:

        - compute the model output, the hidden states, and the other states
        - compute the hidden-to-weight Jacobian and the hidden-to-hidden Jacobian
        - compute the loss-to-hidden or loss-to-weight Jacobian

        Then, ``self._update_etrace_data`` is called to:

        - update the eligibility trace data

        The forward function returns two parts of data:

        - The first part is the functional returns (same as "self._update()" function):
              * the model output
              * the updated hidden states
              * the updated other states
              * the updated eligibility trace states

        - The second part is the data used for backward gradient computation:
              * the residuals of the model
              * the eligibility trace data at the current/last time step
              * the weight id to its value mapping
              * the running index
        """

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, othstate_vals, write=False)

        # necessary gradients of the weights
        (
            out,
            hiddens,
            oth_states,
            hid2weight_jac_multi_steps,
            hid2hid_jac_multi_steps,
            residuals
        ) = self.graph.solve_h2w_h2h_jacobian_and_l2h_vjp(*args)

        # eligibility trace update
        #
        # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        new_etrace_vals = self._update_etrace_data(
            running_index,
            etrace_vals,
            hid2weight_jac_multi_steps,
            hid2hid_jac_multi_steps,
            weight_vals,
        )

        # returns
        old_etrace_vals = etrace_vals
        fwd_out = (out, hiddens, oth_states, new_etrace_vals)
        fwd_res = (
            residuals,
            (
                old_etrace_vals
                if self.graph.is_multi_step_vjp else
                new_etrace_vals
            ),
            weight_vals,
            running_index
        )
        return fwd_out, fwd_res

    def _update_fn_bwd(
        self,
        fwd_res,
        grads,
    ) -> Tuple[dG_Inputs, dG_Weight, dG_Hidden, dG_State, None, None]:
        """
        The backward function to compute the VJP gradients when the learning signal is arrived at
        this time step.

        There are three steps:

        1. Interpret the forward results (eligibility trace) and top-down gradients (learning signal)
        2. Compute the gradients of input arguments
           (maybe necessary, but it can be optimized away but the XLA compiler)
        3. Compute the gradients of the weights

        """

        # [1] Interpret the fwd results
        #
        (
            residuals,  # the residuals of the VJP computation, for computing the gradients of input arguments
            etrace_vals_at_t_or_t_minus_1,  # the eligibility trace data at the current or last time step
            weight_vals,  # the weight id to its value mapping
            running_index  # the running index
        ) = fwd_res

        (
            jaxpr,
            in_tree,
            out_tree,
            consts
        ) = residuals

        # [2] Interpret the top-down gradient signals
        #
        # Since
        #
        #     dg_out, dg_hiddens, dg_others, dg_etrace = grads
        #
        # we need to remove the "dg_etrace" iterm from the gradients for matching
        # the jaxpr vjp gradients.
        #
        grad_flat, grad_tree = jax.tree.flatten((grads[:-1],))

        # [3] Compute the gradients of the input arguments
        #     It may be unnecessary, but it can be optimized away by the XLA compiler after it is computed.
        #
        # The input argument gradients are computed through the normal back-propagation algorithm.
        #
        if out_tree != grad_tree:
            raise TypeError(
                f'Gradient tree should be the same as the function output tree. '
                f'While we got: \n'
                f'out_tree  = {out_tree}\n!=\n'
                f'grad_tree = {grad_tree}'
            )
        cts_out = jax.core.eval_jaxpr(jaxpr, consts, *grad_flat)

        #
        # We compute:
        #
        #   - the gradients of input arguments,
        #     maybe necessary to propagate the gradients to the last layer
        #
        #   - the gradients of the hidden states at the last time step,
        #     maybe unnecessary but can be optimized away by the XLA compiler
        #
        #   - the gradients of the non-etrace parameters, defined by "NonTempParam"
        #
        #   - the gradients of the other states
        #
        #   - the gradients of the loss-to-hidden at the current time step
        #

        # the `_jaxpr_compute_model_with_vjp()` in `ETraceGraphExecutor`
        (
            dg_args,
            dg_last_hiddens,
            dg_non_etrace_params,
            dg_etrace_params,
            dg_oth_states,
            dl_to_dh_at_t
        ) = jax.tree.unflatten(in_tree, cts_out)

        if self.graph.is_single_step_vjp:

            # TODO: the correspondence between the hidden states and the gradients
            #        should be checked.
            #
            assert len(self.compiled.out_hidden_jaxvars) == len(dl_to_dh_at_t)
            for hid_var, dg in zip(self.compiled.out_hidden_jaxvars, dl_to_dh_at_t):
                assert hid_var.aval.shape == dg.shape, (
                    'The shape of the hidden states and its gradients should be the same. '
                )
            dl_to_dh_at_t_or_t_minus_1 = {
                self.compiled.hid_outvar_to_path[hid_var]: dg
                for hid_var, dg in
                zip(self.compiled.out_hidden_jaxvars, dl_to_dh_at_t)
            }
            assert len(dg_etrace_params) == 0  # gradients all etrace weights are updated by the RTRL algorithm

        else:

            assert len(dg_last_hiddens) == len(self.hidden_states)
            assert set(dg_last_hiddens.keys()) == set(self.hidden_states.keys()), (
                f'The hidden states should be the same. '
            )
            dl_to_dh_at_t_or_t_minus_1 = dg_last_hiddens

        #
        # [4] Compute the gradients of the weights
        #
        # the gradients of the weights are computed through the RTRL algorithm.
        #
        # "self._solve_weight_gradients()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        dg_weights = self._solve_weight_gradients(
            running_index,
            etrace_vals_at_t_or_t_minus_1,
            dl_to_dh_at_t_or_t_minus_1,
            weight_vals,
            dg_non_etrace_params,
            dg_etrace_params,
        )

        # Note that there are no gradients flowing through the etrace data and the running index.
        dg_etrace = None
        dg_running_index = None

        return (
            dg_args,
            dg_weights,
            dg_last_hiddens,
            dg_oth_states,
            dg_etrace,
            dg_running_index
        )

    def _solve_weight_gradients(
        self,
        running_index: Optional[int],
        etrace_h2w_at_t: Any,
        dl_to_dh_at_t: Sequence[jax.Array],
        weight_vals: Dict[WeightID, PyTree],
        dl_to_nonetws_at_t: List[PyTree],
        dl_to_etws_at_t: Optional[List[PyTree]],
    ):
        r"""
        The method to solve the weight gradients, i.e., :math:`\partial L / \partial W`.

        .. note::

            This is the protocol method that should be implemented in the subclass.


        Particularly, the weight gradients are computed through::

        .. math::

            \frac{\partial L^t}{\partial W} = \frac{\partial L^t}{\partial h^t} \frac{\partial h^t}{\partial W}

        Or,

        .. math::

            \frac{\partial L^t}{\partial W} = \frac{\partial L^{t-1}}{\partial h^{t-1}}
                                              \frac{\partial h^{t-1}}{\partial W}
                                              + \frac{\partial L^t}{\partial W^t}


        Args:
          running_index: Optional[int], the running index.
          etrace_h2w_at_t: Any, the eligibility trace data (which track the hidden-to-weight Jacobian)
              that have accumulated util the time ``t``.
          dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array], the gradients of the loss-to-hidden
              at the time ``t``.
          weight_vals: Dict[WeightID, PyTree], the weight values.
          dl_to_nonetws_at_t: List[PyTree], the gradients of the loss-to-non-etrace parameters
              at the time ``t``, i.e., :math:``\partial L^t / \partial W^t``.
          dl_to_etws_at_t: List[PyTree], the gradients of the loss-to-etrace parameters
              at the time ``t``, i.e., :math:``\partial L^t / \partial W^t``.
        """
        raise NotImplementedError

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        etrace_vals_util_t_1: ETraceVals,
        hid2weight_jac_multi_times: Hid2WeightJacobian,
        hid2hid_jac_multi_times: Sequence[jax.Array],
        weight_vals: WeightVals,
    ) -> ETraceVals:
        """
        The method to update the eligibility trace data.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
          running_index: Optional[int], the running index.
          etrace_vals_util_t_1: ETraceVals, the history eligibility trace data that have accumulated util :math:`t-1`.
          hid2weight_jac_multi_times: ETraceVals, the current eligibility trace data at the time :math:`t`.
          hid2hid_jac_multi_times: The data for computing the hidden-to-hidden Jacobian at the time :math:`t`.
          weight_vals: Dict[WeightID, PyTree], the weight values.

        Returns:
          ETraceVals, the updated eligibility trace data that have accumulated util :math:`t`.
        """
        raise NotImplementedError

    def _get_etrace_data(self) -> ETraceVals:
        """
        Get the eligibility trace data at the last time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Returns:
          ETraceVals, the eligibility trace data.
        """
        raise NotImplementedError

    def _assign_etrace_data(self, etrace_vals: ETraceVals) -> None:
        """
        Assign the eligibility trace data to the states at the current time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
          etrace_vals: ETraceVals, the eligibility trace data.
        """
        raise NotImplementedError


# class TruncatedVjpAlgorithm(ETraceVjpAlgorithm):
#     r"""
#     The online gradient computation algorithm with the diagonal approximation and
#     the input-output dimensional complexity.
#
#     This algorithm has the :math:`O(nBI+nBO)` memory complexity and :math:`O(nBIO)` computational
#     complexity, where :math:`I` and :math:`O` are the number of input and output dimensions,
#     :math:`B` the batch size, and :math:`n` the number of truncation length.
#
#     Parameters:
#     -----------
#     model: brainstate.nn.Module
#         The model function, which receives the input arguments and returns the model output.
#     vjp_time: str, optional
#         The time to compute the loss-to-hidden Jacobian.
#
#         - ``0``: the current time step: $\frac{\partial L^t}{\partial h^t}$.  Memory is
#         - ``1``: the last time step: $\frac{\partial L^{t-1}}{\partial h^{t-1}}$.
#         - ``k``: the t-k time step: $\frac{\partial L^{t-k}}{\partial h^{t-k}}$.
#     name: str, optional
#         The name of the etrace algorithm.
#     n_truncation: int
#         The truncation length.
#     """
#
#     __module__ = 'brainscale'
#
#     # the spatial gradients of the weights
#     etrace_xs: Dict[WeightXVar, bst.State]
#
#     # the spatial gradients of the hidden states
#     etrace_dfs: Dict[Tuple[WeightYVar, HiddenOutVar], bst.State]
#
#     # the truncation length
#     n_truncation: int
#
#     def __init__(
#         self,
#         model: bst.nn.Module,
#         n_truncation: int,
#         name: Optional[str] = None,
#         vjp_method: str = 'single-step',
#     ):
#         super().__init__(model, name=name, vjp_method=vjp_method)
#
#         # the learning parameters
#         assert isinstance(n_truncation, int), 'The truncation length should be an integer. '
#         assert n_truncation > 0, 'The truncation length should be greater than 0. '
#         self.n_truncation = n_truncation
#
#     def init_etrace_state(self, *args, **kwargs):
#         """
#         Initialize the eligibility trace states of the etrace algorithm.
#
#         This method is needed after compiling the etrace graph. See
#         `.compile_graph()` for the details.
#         """
#         # The states of weight spatial gradients:
#         #   1. x
#         #   2. df
#         self.etrace_xs = dict()
#         self.etrace_dfs = dict()
#         for relation in self.compiled.hidden_param_op_relations:
#             # For the relation
#             #
#             #   h1, h2, ... = f(x, w)
#             #
#             # we need to initialize the eligibility trace states for the weight x and the df.
#
#             if relation.x not in self.etrace_xs:
#                 shape = (self.n_truncation,) + relation.x.aval.shape
#                 dtype = relation.x.aval.dtype
#                 # wx may be repeatedly used in the graph
#                 self.etrace_xs[relation.x] = EligibilityTraceData(u.math.zeros(shape, dtype))
#
#             for group in relation.hidden_groups:
#                 group: HiddenGroup
#                 #
#                 # Group 1:
#                 #
#                 #   [∂a^t-1/∂θ1, ∂b^t-1/∂θ1, ...]
#                 #
#                 # Group 2:
#                 #
#                 #   [∂A^t-1/∂θ1, ∂B^t-1/∂θ1, ...]
#                 #
#                 for hidden_outvar in group.hidden_outvars:
#                     key = (relation.y, hidden_outvar)
#                     if key in self.etrace_dfs:  # relation.y is an unique output of the weight operation
#                         raise ValueError(f'The relation {key} has been added. ')
#                     shape = (self.n_truncation,) + relation.y.aval.shape
#                     dtype = relation.y.aval.dtype
#                     self.etrace_dfs[key] = EligibilityTraceData(u.math.zeros(shape, dtype))
#
#     def reset_state(self, batch_size: int = None, **kwargs):
#         """
#         Reset the eligibility trace states.
#         """
#
#         def _zeros_like_batch_(x: jax.Array):
#             if batch_size is not None:
#                 assert isinstance(batch_size, int), 'The batch size should be an integer. '
#                 shape = (self.n_truncation, batch_size) + x.shape[1:]
#             else:
#                 shape = (self.n_truncation,) + x.shape
#             return u.math.zeros(shape, x.dtype)
#
#         for k, v in self.etrace_xs.items():
#             self.etrace_xs[k].value = jax.tree.map(_zeros_like_batch_, v)
#         for k, v in self.etrace_dfs.items():
#             self.etrace_dfs[k].value = jax.tree.map(_zeros_like_batch_, v)
#
#     def _get_etrace_data(self) -> Tuple:
#         etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
#         etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
#         return etrace_xs, etrace_dfs
#
#     def _assign_etrace_data(self, hist_etrace_vals):
#         #
#         # For any operation:
#         #
#         #           h^t = f(x^t \theta)
#         #
#         # etrace_xs:
#         #           x^t
#         #
#         # etrace_dfs:
#         #           df^t = ∂h^t / ∂y^t, where y^t = x^t \theta
#         #
#         (etrace_xs, etrace_dfs) = hist_etrace_vals
#
#         # the weight x and df
#         for x, val in etrace_xs.items():
#             self.etrace_xs[x].value = val
#         for dfkey, val in etrace_dfs.items():
#             self.etrace_dfs[dfkey].value = val
#
#     def _update_etrace_data(
#         self,
#         running_index: Optional[int],
#         hist_etrace_vals: PyTree,
#         hid2weight_jac_multi_times: Hid2WeightJacobian,
#         hid2hid_jac_multi_times: Hid2HidJacobian,
#         weight_vals: Dict[Path, PyTree],
#     ) -> ETraceVals:
#         #
#         # "running_index":
#         #            the running index
#         #
#         # "hist_etrace_vals":
#         #            the history etrace values,
#         #            including the x and df values, see "etrace_xs" and "etrace_dfs".
#         #
#         # "hid2weight_jac_multi_times":
#         #           the current etrace values at the time "t", \epsilon^t, if vjp_time == "t".
#         #           Otherwise, the etrace values at the time "t-1", \epsilon^{t-1}.
#         #
#         # "hid2hid_jac_multi_times":
#         #           the data for computing the hidden-to-hidden Jacobian at the time "t".
#         #
#         # "weight_path_to_vals":
#         #           the weight values.
#         #
#
#         # --- the data --- #
#         #
#         # the etrace data at the current time step (t) of the O(n) algorithm
#         # is a tuple, including the weight x and df values.
#         #
#         # For the weight x, it is a dictionary,
#         #    {WeightXVar: jax.Array}
#         #
#         # For the weight df, it is a dictionary,
#         #    {(WeightYVar, HiddenOutVar): jax.Array}
#         #
#         xs, dfs = hid2weight_jac_multi_times
#
#         #
#         # the history etrace values
#         #
#         # - hist_xs: {WeightXVar: brainstate.State}
#         # - hist_dfs: {(WeightYVar, HiddenOutVar): brainstate.State}
#         #
#         hist_xs, hist_dfs = hist_etrace_vals
#
#         #
#         # the new etrace values
#         new_etrace_xs, new_etrace_dfs = dict(), dict()
#
#         # --- the update --- #
#
#         #
#         # Step 1:
#         #
#         #   update the weight x using the equation:
#         #           x^t = α * x^t-1 + x^t, where α is the decay factor.
#         #
#         for xkey in hist_xs.keys():
#             new_etrace_xs[xkey] = u.math.concatenate((hist_xs[xkey][1:], u.math.expand_dims(xs[xkey], 0)), axis=0)
#
#         for hwo_relation in self.compiled.hidden_param_op_relations:
#             hwo_relation: HiddenParamOpRelation
#
#             for group in hwo_relation.hidden_groups:
#                 group: HiddenGroup
#                 #
#                 # Step 2:
#                 #
#                 # update the eligibility trace * hidden diagonal Jacobian
#                 #         dϵ^t = D_h ⊙ dϵ^t-1, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
#                 #
#                 #
#                 # JVP equation for the following Jacobian computation:
#                 #
#                 # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ1,
#                 #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ1,]
#                 #
#                 # [∂V^t/∂V^t-1, ∂V^t/∂a^t-1,  [∂V^t-1/∂θ2,
#                 #  ∂a^t/∂V^t-1, ∂a^t/∂a^t-1]   ∂a^t-1/∂θ2]
#                 #
#
#                 for hidden_outvar_at_t in group.hidden_outvars:
#                     #
#                     # computing the following vector-Jacobian product:
#                     #  df^t = D_h ⊙ df^t-1
#                     #
#                     # i.e., the hidden-to-hidden Jacobian diagonal matrix * the hidden df at the previous time step
#                     #
#                     #  ∂V^t/∂θ1 = ∂V^t/∂V^t-1 * ∂V^t-1/∂θ1 + ∂V^t/∂a^t-1 * ∂a^t-1/∂θ1 + ...
#                     #
#                     new_etrace = None
#                     for hidden_outvar_at_t_minus_1 in group.hidden_outvars:
#                         diag_jac_key = (hidden_outvar_at_t_minus_1, hidden_outvar_at_t)
#                         if diag_jac_key in hid2hid_jac_multi_times:
#                             # diagonal Jacobian * hidden df
#                             data = (hist_dfs[(hwo_relation.y, hidden_outvar_at_t_minus_1)] *
#                                     u.math.expand_dims(hid2hid_jac_multi_times[diag_jac_key], axis=0))
#                             new_etrace = data if new_etrace is None else new_etrace + data
#                     assert new_etrace is not None, f'The new etrace should not be None. '
#
#                     #
#                     # Step 3:
#                     #
#                     # update: eligibility trace * hidden diagonal Jacobian + new hidden df
#                     #        dϵ^t = D_h ⊙ dϵ^t-1 + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
#                     #
#                     new_df_key = (hwo_relation.y, hidden_outvar_at_t)
#                     new_etrace_dfs[new_df_key] = u.math.concatenate(
#                         [new_etrace[1:],
#                          u.math.expand_dims(dfs.get(new_df_key, u.math.zeros_like(new_etrace[0])), axis=0)],
#                         axis=0
#                     )
#         return new_etrace_xs, new_etrace_dfs
#
#     def _solve_weight_gradients(
#         self,
#         running_index: int,
#         etrace_h2w_at_t: Tuple,
#         dl_to_dh_at_t: Dict[HiddenOutVar, jax.Array],
#         weight_id_to_its_val: Dict[WeightID, PyTree],
#         dl_to_nonetws_at_t: List[PyTree],
#         dl_to_etws_at_t: Optional[List[PyTree]],
#     ):
#         """
#         See the documentation in the super class for the details.
#         """
#
#         dG_weights = {id(st): None for st in self.param_states}
#
#         # update the etrace parameters
#         xs, dfs = etrace_h2w_at_t
#         for relation in self.graph.hidden_param_op_relations:
#             x = xs[relation.x]
#             weight_id = id(relation.weight)
#
#             #
#             # Function to compute the weight gradients
#             # according to the inputs and df gradients
#             #
#             fun_dxy2dw = lambda dx, dy: _weight_op_gradient(relation.op_jaxpr, dx, weight_id_to_its_val[weight_id], dy)
#
#             #
#             # Solve the weight gradients by using the etrace data
#             #
#             #   dw = (dL/dH \circ df) \otimes x
#             #
#             for i, hid_var in enumerate(relation.hidden_vars):
#                 df = dfs[(relation.y, hid_var)]  # the hidden gradients
#                 df_hid = df * u.math.expand_dims(dl_to_dh_at_t[hid_var], axis=0)  # the hidden gradients
#                 #
#                 # Compute the weight gradients according to the x and y
#                 #
#                 #    dw = df(dx, dy)
#                 #
#                 dg_weight = jax.tree.map(lambda a: a.sum(0), jax.vmap(fun_dxy2dw)(x, df_hid))
#                 _update_dict(dG_weights, weight_id, dg_weight)  # update the weight gradients
#
#         # update the non-etrace parameters
#         etrace_params, _, non_etrace_params, _ = split_states_v2(self.graph.states)
#         for st, dg in zip(non_etrace_params, dl_to_nonetws_at_t):
#             _update_dict(dG_weights, id(st), dg)
#
#         # update the etrace parameters when "dl_to_etws_at_t" is not None
#         if dl_to_etws_at_t is not None:
#             for st, dg in zip(etrace_params, dl_to_etws_at_t):
#                 _update_dict(dG_weights, id(st), dg, error_when_no_key=True)
#         return list(dG_weights.values())


class IODimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""
    The online gradient computation algorithm with the diagonal approximation
    and the input-output dimensional complexity.

    This algrithm computes the gradients of the weights with the diagonal approximation
    and the input-output dimensional complexity. Its aglritm is based on the RTRL algorithm,
    and has the following learning rule:

    $$
    \begin{aligned}
    & \boldsymbol{\epsilon}^t \approx \boldsymbol{\epsilon}_{\mathbf{f}}^t \otimes \boldsymbol{\epsilon}_{\mathbf{x}}^t \\
    & \boldsymbol{\epsilon}_{\mathbf{x}}^t=\alpha \boldsymbol{\epsilon}_{\mathbf{x}}^{t-1}+\mathbf{x}^t \\
    & \boldsymbol{\epsilon}_{\mathbf{f}}^t=\alpha \operatorname{diag}\left(\mathbf{D}^t\right) \circ \boldsymbol{\epsilon}_{\mathbf{f}}^{t-1}+(1-\alpha) \operatorname{diag}\left(\mathbf{D}_f^t\right) \\
    & \nabla_{\boldsymbol{\theta}} \mathcal{L}=\sum_{t^{\prime} \in \mathcal{T}} \frac{\partial \mathcal{L}^{t^{\prime}}}{\partial \mathbf{h}^{t^{\prime}}} \circ \boldsymbol{\epsilon}^{t^{\prime}}
    \end{aligned}
    $$

    For more details, please see `the ES-D-RTRL algorithm presented in our manuscript <https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2>`_.

    This algorithm has the :math:`O(BI+BO)` memory complexity and :math:`O(BIO)` computational
    complexity, where :math:`I` and :math:`O` are the number of input and output dimensions, and
    :math:`B` the batch size.

    Particularly, for a Linear transformation layer, the algorithm computes the weight gradients
    with the :math:`O(Bn)` memory complexity and :math:`O(Bn^2)` computational complexity, where
    :math:`n` is the number of hidden dimensions.

    Parameters:
    -----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    vjp_time: str, optional
        The time to compute the loss-to-hidden Jacobian.

        - ``0``: the current time step: $\frac{\partial L^t}{\partial h^t}$.  Memory is
        - ``1``: the last time step: $\frac{\partial L^{t-1}}{\partial h^{t-1}}$.
        - ``k``: the t-k time step: $\frac{\partial L^{t-k}}{\partial h^{t-k}}$.

    decay_or_rank: float, int
        The exponential smoothing factor for the eligibility trace. If it is a float,
        it is the decay factor, should be in the range of (0, 1). If it is an integer,
        it is the number of approximation rank for the algorithm, should be greater than 0.
    name: str, optional
        The name of the etrace algorithm.
    mode: Optional[brainscale.mixin.Mode]
        The computing mode, indicating the batching information.
    """

    __module__ = 'brainscale'

    # the spatial gradients of the weights
    etrace_xs: Dict[ETraceX_Key, bst.State]

    # the spatial gradients of the hidden states
    etrace_dfs: Dict[ETraceDF_Key, bst.State]

    # the mapping from the etrace x to the weight operations
    etrace_xs_to_weights = Dict[ETraceX_Key, List[Path]]

    # the exponential smoothing decay factor
    decay: float

    def __init__(
        self,
        model: bst.nn.Module,
        decay_or_rank: float | int,
        mode: Optional[bst.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

        # computing mode
        self.mode = bst.mixin.Mode() if mode is None else mode

        # the learning parameters
        self.decay, num_rank = _format_decay_and_rank(decay_or_rank)

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See :meth:`.compile_graph()` for the details.
        """
        # The states of weight spatial gradients:
        #   1. x
        #   2. df
        self.etrace_xs = dict()
        self.etrace_dfs = dict()
        self.etrace_xs_to_weights = defaultdict(list)
        for relation in self.compiled.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            _init_IO_dim_state(
                self.etrace_xs,
                self.etrace_dfs,
                self.etrace_xs_to_weights,
                self.graph.state_id_to_path,
                relation,
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_xs, batch_size)
        _reset_state_in_a_dict(self.etrace_dfs, batch_size)

    # def get_etrace_of(self, weight: bst.ParamState | Path) -> Tuple[Dict, Dict]:
    #     """
    #     Get the eligibility trace of the given weight.
    #
    #     The eligibility trace contains the following structures:
    #
    #     """
    #     self._assert_compiled()
    #
    #     # the weight ID
    #     weight_id = (
    #         id(weight)
    #         if isinstance(weight, bst.ParamState) else
    #         id(self.graph.path_to_state[weight])
    #     )
    #
    #     etrace_xs = dict()
    #     etrace_dfs = dict()
    #     find_this_weight = False
    #     for relation in self.compiled.hidden_param_op_relations:
    #         relation: HiddenParamOpRelation
    #         if id(relation.weight) != weight_id:
    #             continue
    #         find_this_weight = True
    #
    #         # get the weight_op input
    #         wx_var = relation.x
    #         etrace_xs[wx_var] = self.etrace_xs[wx_var].value
    #
    #         # get the weight_op df
    #         wy_var = relation.y
    #         for group in relation.hidden_groups:
    #             group: HiddenGroup
    #             for st in group.hidden_states:
    #                 path = self.state_id_to_path[id(st)]
    #                 etrace_dfs[(wy_var, path)] = self.etrace_dfs[(wy_var, path)].value
    #     if not find_this_weight:
    #         raise ValueError(f'Do not the etrace of the given weight: {weight}.')
    #     return etrace_xs, etrace_dfs

    def _get_etrace_data(self) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        etrace_xs = {k: v.value for k, v in self.etrace_xs.items()}
        etrace_dfs = {k: v.value for k, v in self.etrace_dfs.items()}
        return etrace_xs, etrace_dfs

    def _assign_etrace_data(
        self,
        hist_etrace_vals: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ]
    ):
        #
        # For any operation:
        #
        #           h^t = f(x^t \theta)
        #
        # etrace_xs:
        #           x^t
        #
        # etrace_dfs:
        #           df^t = ∂h^t / ∂y^t, where y^t = x^t \theta
        #
        (etrace_xs, etrace_dfs) = hist_etrace_vals

        # the weight x and df
        for x, val in etrace_xs.items():
            self.etrace_xs[x].value = val
        for df, val in etrace_dfs.items():
            self.etrace_dfs[df].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ],
        hid2weight_jac_multi_times: Hid2WeightJacobian,
        hid2hid_jac_multi_times: HidGroupJacobian,
        weight_vals: WeightVals,
    ) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        #
        # "running_index":
        #            the running index
        #
        # "hist_etrace_vals":
        #            the history etrace values,
        #            including the x and df values, see "etrace_xs" and "etrace_dfs".
        #
        # "hid2weight_jac_multi_times":
        #           the current etrace values at the time "t", \epsilon^t, if vjp_time == "t".
        #           Otherwise, the etrace values at the time "t-1", \epsilon^{t-1}.
        #
        # "hid2hid_jac_multi_times":
        #           the data for computing the hidden-to-hidden Jacobian at the time "t".
        #
        # "weight_path_to_vals":
        #           the weight values.
        #

        scan_fn = partial(
            _update_IO_dim_etrace_scan_fn,
            hid_weight_op_relations=self.compiled.hidden_param_op_relations,
            decay=self.decay,
        )
        hist_etrace_vals, _ = jax.lax.scan(
            scan_fn,
            hist_etrace_vals,
            (
                hid2weight_jac_multi_times[0],
                hid2weight_jac_multi_times[1],
                hid2hid_jac_multi_times,
            ),
        )
        return hist_etrace_vals

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ],
        dl_to_dh_at_t: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """
        See the documentation in the super class for the details.
        """

        #
        # dl_to_dh_at_t:
        #         The gradients of the loss-to-hidden-group at the time "t".
        #         It has the shape of [n_hidden, ..., n_state].
        #         - `l` is the loss,
        #         - `h` is the hidden group,
        #
        # dl_to_nonetws_at_t:
        #         The gradients of the loss-to-non-etrace parameters
        #         at the time "t", i.e., ∂L^t / ∂W^t.
        #         It has the shape of [n_param, ...].
        #
        # dl_to_etws_at_t:
        #         The gradients of the loss-to-etrace parameters
        #         at the time "t", i.e., ∂L^t / ∂W^t.
        #         It has the shape of [n_param, ...].
        #
        dG_weights = {path: None for path in self.param_states.keys()}

        # update the etrace parameters
        _solve_IO_dim_weight_gradients(
            etrace_h2w_at_t,
            dG_weights,
            dl_to_dh_at_t,
            self.compiled.hidden_param_op_relations,
            running_index,
            self.decay,
        )

        # update the non-etrace parameters
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # update the etrace parameters when "dl_to_etws_at_t" is not None
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights


class ParamDimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""
    The online gradient computation algorithm with the diagonal approximation and the parameter dimension complexity.

    This algorithm computes the gradients of the weights with the diagonal approximation and the parameter dimension complexity.
    Its algorithm is based on the RTRL algorithm, and has the following learning rule:

    $$
    \begin{aligned}
    &\boldsymbol{\epsilon}^t \approx \mathbf{D}^t \boldsymbol{\epsilon}^{t-1}+\operatorname{diag}\left(\mathbf{D}_f^t\right) \otimes \mathbf{x}^t \\
    & \nabla_{\boldsymbol{\theta}} \mathcal{L}=\sum_{t^{\prime} \in \mathcal{T}} \frac{\partial \mathcal{L}^{t^{\prime}}}{\partial \mathbf{h}^{t^{\prime}}} \circ \boldsymbol{\epsilon}^{t^{\prime}}
    \end{aligned}
    $$

    For more details, please see `the D-RTRL algorithm presented in our manuscript <https://www.biorxiv.org/content/10.1101/2024.09.24.614728v2>`_.

    Note than the :py:class:`ParamDimVjpAlgorithm` is a subclass of :py:class:`brainstate.nn.Module`,
    and it is sensitive to the context/mode of the computation. Particularly,
    the :py:class:`ParamDimVjpAlgorithm` is sensitive to ``brainstate.mixin.Batching`` behavior.

    This algorithm has the :math:`O(B\theta)` memory complexity, where :math:`\theta` is the number of parameters,
    and :math:`B` the batch size.

    For a convolutional layer, the algorithm computes the weight gradients with the :math:`O(B\theta)`
    memory complexity, where :math:`\theta` is the dimension of the convolutional kernel.

    For a Linear transformation layer, the algorithm computes the weight gradients with the :math:`O(BIO)``
    computational complexity, where :math:`I` and :math:`O` are the number of input and output dimensions.

    Parameters:
    -----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    vjp_time: str, optional
        The time to compute the loss-to-hidden Jacobian.

        - ``0``: the current time step: $\frac{\partial L^t}{\partial h^t}$.  Memory is
        - ``1``: the last time step: $\frac{\partial L^{t-1}}{\partial h^{t-1}}$.
        - ``k``: the t-k time step: $\frac{\partial L^{t-k}}{\partial h^{t-k}}$.
    name: str, optional
        The name of the etrace algorithm.
    mode: Optional[brainstate.mixin.Mode]
        The computing mode, indicating the batching behavior.
    """

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, bst.State]

    def __init__(
        self,
        model: bst.nn.Module,
        mode: Optional[bst.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)
        self.mode = bst.mixin.Mode() if mode is None else mode

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        `.compile_graph()` for the details.
        """
        # The states of batched weight gradients
        self.etrace_bwg = dict()
        for relation in self.compiled.hidden_param_op_relations:
            _init_param_dim_state(
                self.mode,
                self.etrace_bwg,
                relation,
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)

    def get_etrace_of(self, weight: bst.ParamState | Path) -> Dict:
        """
        Get the eligibility trace of the given weight.

        The eligibility trace contains the following structures:

        """

        self._assert_compiled()

        # get the wight id
        weight_id = (
            id(weight)
            if isinstance(weight, bst.ParamState) else
            id(self.graph.path_to_states[weight])
        )

        find_this_weight = False
        etraces = dict()
        for relation in self.compiled.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            if id(relation.weight) != weight_id:
                continue
            find_this_weight = True

            # retrieve the etrace data
            for group in relation.hidden_groups:
                group: HiddenGroup
                key = etrace_param_key(relation.path, relation.y, group.index)
                etraces[key] = self.etrace_bwg[key].value

        if not find_this_weight:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')
        return etraces

    def _get_etrace_data(self) -> Dict:
        return {
            k: v.value
            for k, v in self.etrace_bwg.items()
        }

    def _assign_etrace_data(self, etrace_vals: Dict) -> None:
        for x, val in etrace_vals.items():
            self.etrace_bwg[x].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Dict[ETraceWG_Key, PyTree],
        hid2weight_jac_multi_times: Hid2WeightJacobian,
        hid2hid_jac_multi_times: HidGroupJacobian,
        weight_vals: Dict[Path, PyTree],
    ) -> Dict[ETraceWG_Key, PyTree]:

        scan_fn = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=self.compiled.hidden_param_op_relations,
            mode=self.mode,
        )

        new_etrace = jax.lax.scan(
            scan_fn,
            hist_etrace_vals,
            (
                hid2weight_jac_multi_times[0],
                hid2weight_jac_multi_times[1],
                hid2hid_jac_multi_times,
            )
        )[0]

        return new_etrace

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree],
        dl_to_dh_at_t: Dict[Path, jax.Array],
        weight_vals: Dict[WeightID, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """
        Solve the weight gradients according to the eligibility trace data.

        Particularly, for each weight, we compute its gradients according to the batched weight gradients.
        """
        dG_weights = {path: None for path in self.param_states}

        # update the etrace weight gradients
        _solve_param_dim_weight_gradients(
            etrace_h2w_at_t,
            dG_weights,
            dl_to_dh_at_t,
            self.compiled.hidden_param_op_relations,
            self.mode,
        )

        # update the non-etrace weight gradients
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # update the etrace parameters when "dl_to_etws_at_t" is not None
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights


class HybridDimVjpAlgorithm(ETraceVjpAlgorithm):
    r"""
    The hybrid online gradient computation algorithm with the diagonal approximation and hybrid complexity.

    Similar to :py:class:`ParamDimVjpAlgorithm`, :py:class:`HybridDimVjpAlgorithm` is a subclass of
    :py:class:`brainstate.nn.Module`, and it is sensitive to the context/mode of the computation.
    Particularly, the :py:class:`ParamDimVjpAlgorithm` is sensitive to ``brainstate.mixin.Batching`` behavior.

    For a function :math:`O = f(I, \theta)`, where :math:`I` is the input, :math:`\theta` is the parameters,
    and :math:`O` is the output, the algorithm computes the weight gradients with the ``O(BI + BO)`` memory complexity
    when :math:`I + O < \theta`, or the ``O(B\theta)`` memory complexity when :math:`I + O \geq \theta`.

    This means that the algorithm combine the memory efficiency of the :py:class:`ParamDimVjpAlgorithm` and the
    computational efficiency of the :py:class:`IODimVjpAlgorithm` together.

    Parameters:
    -----------
    model: Callable
        The model function, which receives the input arguments and returns the model output.
    vjp_time: str, optional
        The time to compute the loss-to-hidden Jacobian.

        - ``0``: the current time step: $\frac{\partial L^t}{\partial h^t}$.  Memory is
        - ``1``: the last time step: $\frac{\partial L^{t-1}}{\partial h^{t-1}}$.
        - ``k``: the t-k time step: $\frac{\partial L^{t-k}}{\partial h^{t-k}}$.
    name: str, optional
        The name of the etrace algorithm.
    decay_or_rank: float, int
        The exponential smoothing factor for the eligibility trace. If it is a float,
        it is the decay factor, should be in the range of (0, 1). If it is an integer,
        it is the number of approximation rank for the algorithm, should be greater than 0.
    mode: Optional[brainstate.mixin.Mode]
        The computing mode, indicating the batching behavior.
    """

    # the spatial gradients of the weights
    etrace_xs: Dict[ETraceX_Key, bst.State]

    # the spatial gradients of the hidden states
    etrace_dfs: Dict[ETraceDF_Key, bst.State]

    # the mapping from the etrace x to the weight operations
    etrace_xs_to_weights = Dict[ETraceX_Key, List[Path]]

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, bst.State]

    # the exponential smoothing decay factor
    decay: float

    def __init__(
        self,
        model: bst.nn.Module,
        decay_or_rank: float | int,
        mode: Optional[bst.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

        # computing mode
        self.mode = bst.mixin.Mode() if mode is None else mode

        # the learning parameters
        self.decay, num_rank = _format_decay_and_rank(decay_or_rank)

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        `.compile_graph()` for the details.
        """
        #
        # The states of weight spatial gradients:
        #   1. x
        #   2. df
        #   3. batched weight gradients
        #
        self.etrace_xs = dict()
        self.etrace_dfs = dict()
        self.etrace_bwg = dict()
        self.etrace_xs_to_weights = defaultdict(list)

        for relation in self.compiled.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                _init_param_dim_state(
                    self.mode,
                    self.etrace_bwg,
                    relation
                )
            else:
                _init_IO_dim_state(
                    self.etrace_xs,
                    self.etrace_dfs,
                    self.etrace_xs_to_weights,
                    self.graph.state_id_to_path,
                    relation,
                )

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_xs, batch_size)
        _reset_state_in_a_dict(self.etrace_dfs, batch_size)
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)

    def get_etrace_of(self, weight: bst.ParamState | Path) -> Tuple[Dict, Dict, Dict]:
        """
        Get the eligibility trace of the given weight.

        The eligibility trace contains the following structures:

        """

        self._assert_compiled()

        # the weight ID
        weight_id = (
            id(weight)
            if isinstance(weight, bst.ParamState) else
            id(self.graph.path_to_states[weight])
        )

        etrace_xs = dict()
        etrace_dfs = dict()
        etrace_bws = dict()
        find_this_weight = False
        for relation in self.compiled.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            if id(relation.weight) != weight_id:
                continue
            find_this_weight = True

            wx_var = relation.x
            if wx_var in self.etrace_xs:
                # get the weight_op input
                etrace_xs[wx_var] = self.etrace_xs[wx_var].value

                # get the weight_op df
                for group in relation.hidden_groups:
                    group: HiddenGroup
                    df_key = etrace_df_key(relation.y, group.index)
                    etrace_dfs[df_key] = self.etrace_dfs[df_key].value

            # get the batched weight gradients
            for group in relation.hidden_groups:
                group: HiddenGroup
                bwg_key = etrace_param_key(relation.path, relation.y, group.index)
                if bwg_key in self.etrace_bwg:
                    etrace_bws[bwg_key] = self.etrace_bwg[bwg_key].value

        if not find_this_weight:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')

        return etrace_xs, etrace_dfs, etrace_bws

    def _get_etrace_data(self) -> Tuple[Dict, ...]:
        etrace_xs = {x: val.value for x, val in self.etrace_xs.items()}
        etrace_dfs = {x: val.value for x, val in self.etrace_dfs.items()}
        etrace_wgrads = {x: val.value for x, val in self.etrace_bwg.items()}
        return etrace_xs, etrace_dfs, etrace_wgrads

    def _assign_etrace_data(self, etrace_vals: Tuple[Dict, ...]) -> None:
        etrace_xs, etrace_dfs, etrace_wgrads = etrace_vals
        for x, val in etrace_xs.items():
            self.etrace_xs[x].value = val
        for x, val in etrace_dfs.items():
            self.etrace_dfs[x].value = val
        for x, val in etrace_wgrads.items():
            self.etrace_bwg[x].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Tuple[Dict, ...],
        hid2weight_jac_multi_times: Hid2WeightJacobian,
        hid2hid_jac_multi_times: Hid2HidJacobian,
        weight_vals: Dict[Path, PyTree],
    ) -> Tuple[Dict, ...]:

        # the history etrace values
        hist_xs, hist_dfs, hist_bwg = hist_etrace_vals

        # ---- separate the etrace gradients into two parts --- #
        #
        #  1. O(n^2) etrace gradients
        #  2. O(n) etrace gradients
        #

        on_weight_hidden_relations = []
        on2_weight_hidden_relations = []
        for relation in self.compiled.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                on2_weight_hidden_relations.append(relation)
            else:
                on_weight_hidden_relations.append(relation)

        # ---- O(n^2) etrace gradients update ---- #

        scan_fn_on2 = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=on2_weight_hidden_relations,
            mode=self.mode,
        )
        new_bwg = jax.lax.scan(
            scan_fn_on2,
            hist_bwg,
            (
                hid2weight_jac_multi_times[0],
                hid2weight_jac_multi_times[1],
                hid2hid_jac_multi_times,
            )
        )[0]

        # ---- O(n) etrace gradients update ---- #

        scan_fn_on = partial(
            _update_IO_dim_etrace_scan_fn,
            hid_weight_op_relations=self.compiled.hidden_param_op_relations,
            decay=self.decay,
        )
        new_xs, new_dfs = jax.lax.scan(
            scan_fn_on,
            (hist_xs, hist_dfs),
            (
                hid2weight_jac_multi_times[0],
                hid2weight_jac_multi_times[1],
                hid2hid_jac_multi_times,
            ),
        )[0]

        return new_xs, new_dfs, new_bwg

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Tuple,
        dl_to_dh_at_t: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """
        Solve the weight gradients according to the eligibility trace data.

        Particularly, for each weight, we compute its gradients according to the batched weight gradients.
        """

        #
        # dl_to_dh_at_t:
        #         The gradients of the loss-to-hidden-group at the time "t".
        #         It has the shape of [n_hidden, ..., n_state].
        #         - `l` is the loss,
        #         - `h` is the hidden group,
        #
        # dl_to_nonetws_at_t:
        #         The gradients of the loss-to-non-etrace parameters
        #         at the time "t", i.e., ∂L^t / ∂W^t.
        #         It has the shape of [n_param, ...].
        #
        # dl_to_etws_at_t:
        #         The gradients of the loss-to-etrace parameters
        #         at the time "t", i.e., ∂L^t / ∂W^t.
        #         It has the shape of [n_param, ...].
        #

        xs, dfs, wgrads = etrace_h2w_at_t
        dG_weights = {path: None for path in self.param_states.keys()}

        # ---- separate the etrace gradients into two parts --- #
        #
        #  1. O(n^2) etrace gradients
        #  2. O(n) etrace gradients
        #

        on_weight_hidden_relations = []
        on2_weight_hidden_relations = []
        for relation in self.compiled.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                on2_weight_hidden_relations.append(relation)
            else:
                on_weight_hidden_relations.append(relation)

        # --- update the etrace weight gradients by the O(n) algorithm --- #

        _solve_IO_dim_weight_gradients(
            (xs, dfs),
            dG_weights,
            dl_to_dh_at_t,
            on_weight_hidden_relations,
            running_index,
            self.decay,
        )

        # --- update the etrace weight gradients by the O(n^2) algorithm --- #

        _solve_param_dim_weight_gradients(
            wgrads,
            dG_weights,
            dl_to_dh_at_t,
            on2_weight_hidden_relations,
            self.mode,
        )

        # update the non-etrace weight gradients
        for path, dg in dl_to_nonetws_at_t.items():
            _update_dict(dG_weights, path, dg)

        # update the etrace parameters when "dl_to_etws_at_t" is not None
        if dl_to_etws_at_t is not None:
            for path, dg in dl_to_etws_at_t.items():
                _update_dict(dG_weights, path, dg, error_when_no_key=True)
        return dG_weights
