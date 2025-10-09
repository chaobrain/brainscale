# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from collections import defaultdict
from functools import partial
from typing import Dict, Tuple, Any, List, Optional, Sequence

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

from ._etrace_algorithms import (
    ETraceAlgorithm,
    EligibilityTrace,
)
from ._etrace_compiler_hid_param_op import HiddenParamOpRelation
from ._etrace_compiler_hidden_group import HiddenGroup
from ._etrace_concepts import (
    ETraceParam,
    ElemWiseParam,
    ETraceGrad,
)
from ._etrace_input_data import has_multistep_data
from ._etrace_operators import ETraceOp
from ._etrace_vjp_graph_executor import ETraceVjpGraphExecutor
from ._misc import (
    check_dict_keys,
    etrace_x_key,
    etrace_param_key,
    etrace_df_key,
)
from ._state_managment import assign_state_values_v2
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
    HiddenGroupJacobian,
    dG_Inputs,
    dG_Weight,
    dG_Hidden,
    dG_State,
)

__all__ = [
    'ETraceVjpAlgorithm',  # the base class for the eligibility trace algorithm with the VJP gradient computation
    'IODimVjpAlgorithm',  # the diagonally approximated algorithm with the input-output dimension complexity
    'ES_D_RTRL',
    'ParamDimVjpAlgorithm',  # the diagonally approximated algorithm with the parameter dimension complexity
    'D_RTRL',
    'HybridDimVjpAlgorithm',  # the diagonally approximated algorithm with hybrid complexity (either I/O or parameter)
]


def _format_decay_and_rank(decay_or_rank) -> Tuple[float, int]:
    """
    Determines the decay factor and the number of approximation ranks based on the input.

    This function takes either a decay factor or a number of approximation ranks as input
    and returns both the decay factor and the number of approximation ranks. If the input
    is a float, it is treated as a decay factor, and the number of ranks is calculated.
    If the input is an integer, it is treated as the number of ranks, and the decay factor
    is calculated.

    Args:
        decay_or_rank (float or int): The decay factor (a float between 0 and 1) or the 
                                      number of approximation ranks (a positive integer).

    Returns:
        Tuple[float, int]: A tuple containing the decay factor and the number of approximation ranks.

    Raises:
        ValueError: If the input is neither a float nor an integer, or if the float is not in the range (0, 1),
                    or if the integer is not greater than 0.
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
    Apply exponential smoothing to update a value.

    This function performs exponential smoothing, which is a technique used to 
    smooth out data by applying a decay factor to the old value and combining it 
    with the new value. If the new value is None, the function returns the old 
    value scaled by the decay factor.

    Args:
        old: The old value to be smoothed.
        new: The new value to be incorporated into the smoothing. If None, only 
             the old value scaled by the decay factor is returned.
        decay: The decay factor, a float between 0 and 1, that determines the 
               weight of the old value in the smoothing process.

    Returns:
        The smoothed value, which is a combination of the old and new values 
        weighted by the decay factor.
    """
    if new is None:
        return decay * old
    return decay * old + (1 - decay) * new


def _low_pass_filter(old, new, alpha):
    """
    Apply a low-pass filter to smooth the transition between old and new values.

    This function implements a simple low-pass filter, which is used to smooth 
    out fluctuations in data by blending the old value with the new value based 
    on a specified filter factor.

    Parameters
    ----------
    old : Any
        The previous value that needs to be smoothed.
    new : Any
        The current value to be incorporated into the smoothing process. If None, 
        the function will return the old value scaled by the filter factor.
    alpha : float
        The filter factor, a value between 0 and 1, that determines the weight 
        of the old value in the smoothing process. A higher alpha gives more 
        weight to the old value, resulting in slower changes.

    Returns
    -------
    Any
        The filtered value, which is a combination of the old and new values 
        weighted by the filter factor.
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
        the_dict[key] = jax.tree.map(
            u.math.add,
            old_value,
            value,
            is_leaf=lambda x: isinstance(x, u.Quantity)
        )


def _batched_zeros_like(
    batch_size: Optional[int],
    num_state: int,  # the number of hidden states
    x: jax.Array  # the input array
):
    """
    Create a batched zeros array with the same shape as the input array, 
    extended by the number of hidden states.

    This function generates a zeros array that matches the shape of the 
    input array `x`, with an additional dimension for the number of hidden 
    states. If a batch size is provided, the zeros array will also include 
    a batch dimension.

    Args:
        batch_size (Optional[int]): The size of the batch. If None, the 
            batch dimension is not included.
        num_state (int): The number of hidden states, which determines the 
            size of the additional dimension in the zeros array.
        x (jax.Array): The input array whose shape is used as a reference 
            for creating the zeros array.

    Returns:
        jax.Array: A zeros array with the same shape as the input array, 
        extended by the number of hidden states, and optionally including 
        a batch dimension.
    """
    if batch_size is None:
        return u.math.zeros((*x.shape, num_state), x.dtype)
    else:
        return u.math.zeros((batch_size, *x.shape, num_state), x.dtype)


def _init_IO_dim_state(
    etrace_xs: Dict[ETraceX_Key, brainstate.State],
    etrace_dfs: Dict[ETraceDF_Key, brainstate.State],
    etrace_xs_to_weights: defaultdict[ETraceX_Key, List[Path]],
    state_id_to_path: Dict[int, Path],
    relation: HiddenParamOpRelation,
    mode: brainstate.mixin.Mode
):
    """
    Initialize the eligibility trace states for input-output dimensions.

    This function sets up the eligibility trace states for the weights and 
    differential functions (df) associated with a given relation. It ensures 
    that the eligibility trace states are initialized for the weight x and 
    the df, and records the target paths of the weight x if it is used 
    repeatedly in the graph.

    Args:
        etrace_xs (Dict[ETraceX_Key, brainstate.State]): A dictionary to store the
            eligibility trace states for the weight x, keyed by ETraceX_Key.
        etrace_dfs (Dict[ETraceDF_Key, brainstate.State]): A dictionary to store the
            eligibility trace states for the differential functions, keyed by 
            ETraceDF_Key.
        etrace_xs_to_weights (defaultdict[ETraceX_Key, List[Path]]): A 
            defaultdict to record the target paths of the weight x, keyed by 
            ETraceX_Key.
        state_id_to_path (Dict[int, Path]): A dictionary mapping state IDs to 
            their corresponding paths.
        relation (HiddenParamOpRelation): The relation object containing 
            information about the weights and hidden groups involved in the 
            computation.

    Raises:
        ValueError: If a relation with the same key has already been added to 
            the eligibility trace states.
    """
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
            etrace_xs[id(relation.x)] = EligibilityTrace(u.math.zeros(shape, dtype))

        # relation.x maybe repeatedly used to feed into the
        # weight operation for transforming the hidden states
        # therefore we record the target paths of the weight x
        #
        etrace_xs_to_weights[id(relation.x)].append(state_id_to_path[id(relation.weight)])

    y_shape = relation.y.aval.shape
    y_dtype = relation.y.aval.dtype
    for group in relation.hidden_groups:
        group: HiddenGroup
        if y_shape != group.varshape:
            if isinstance(relation.weight, ElemWiseParam):
                if not (mode.is_a(brainstate.mixin.Batching) and y_shape == group.varshape[1:]):
                    raise ValueError(
                        f'The shape of the hidden states should be the '
                        f'same as the shape of the hidden group. '
                        f'While we got {y_shape} != {group.varshape}. '
                    )
            else:
                raise ValueError(
                    f'The shape of the hidden states should be the '
                    f'same as the shape of the hidden group. '
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
        shape = group.varshape + (group.num_state,)
        etrace_dfs[key] = EligibilityTrace(u.math.zeros(shape, y_dtype))


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
    """
    Update the eligibility trace values for input-output dimensions.

    This function updates the eligibility trace values for the weight x and 
    differential functions (df) based on the provided Jacobians and decay 
    factor. It computes the new eligibility trace values by applying a 
    low-pass filter to the historical values and incorporating the current 
    Jacobian values.

    Args:
        hist_etrace_vals (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]):
            A tuple containing dictionaries of historical eligibility trace 
            values for the weight x and df, keyed by ETraceX_Key and 
            ETraceDF_Key, respectively.
        jacobians (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array], Sequence[jax.Array]]):
            A tuple containing dictionaries of current Jacobian values for the 
            weight x and df, and a sequence of hidden group Jacobians.
        hid_weight_op_relations (Sequence[HiddenParamOpRelation]):
            A sequence of HiddenParamOpRelation objects representing the 
            relationships between hidden parameters and operations.
        decay (float): The decay factor used in the low-pass filter, a value 
            between 0 and 1.

    Returns:
        Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]:
            A tuple containing dictionaries of updated eligibility trace values 
            for the weight x and df, keyed by ETraceX_Key and ETraceDF_Key, 
            respectively.
    """
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
    weight_vals: Dict[Path, WeightVals],
    running_index: int,
    decay: float,
    mode: brainstate.mixin.Mode,
):
    """
    Compute and update the weight gradients for input-output dimensions using eligibility trace data.

    This function calculates the weight gradients by utilizing the eligibility trace data and the 
    hidden-to-hidden Jacobians. It applies a correction factor to avoid exponential smoothing bias 
    at the beginning of the computation.

    Args:
        hist_etrace_data (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]):
            A tuple containing dictionaries of historical eligibility trace values for the weight x 
            and differential functions (df), keyed by ETraceX_Key and ETraceDF_Key, respectively.
        dG_weights (Dict[Path, dG_Weight]):
            A dictionary to store the computed weight gradients, keyed by the path of the weight.
        dG_hidden_groups (Sequence[jax.Array]):
            A sequence of hidden group Jacobians, with the same length as the total number of hidden groups.
        weight_hidden_relations (Sequence[HiddenParamOpRelation]):
            A sequence of HiddenParamOpRelation objects representing the relationships between hidden 
            parameters and operations.
        weight_vals (Dict[Path, WeightVals]):
            A dictionary containing the current values of the weights, keyed by their paths.
        running_index (int):
            The current index in the running sequence, used to compute the correction factor.
        decay (float):
            The decay factor used in the exponential smoothing process, a value between 0 and 1.

    Returns:
        None: The function updates the dG_weights dictionary in place with the computed weight gradients.
    """
    # Avoid the exponential smoothing bias at the beginning.
    # This is the correction factor for the exponential smoothing.
    correction_factor = 1. - u.math.power(1. - decay, running_index + 1)
    correction_factor = u.math.where(running_index < 1000, correction_factor, 1.)
    correction_factor = jax.lax.stop_gradient(correction_factor)

    xs, dfs = hist_etrace_data

    for relation in weight_hidden_relations:
        relation: HiddenParamOpRelation

        if not isinstance(relation.weight, ElemWiseParam):
            x = xs[id(relation.x)]
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
            fn_vmap = jax.vmap(
                lambda df: weight_op.xy_to_dw(x, df, weight_vals[weight_path]), in_axes=-1, out_axes=-1,
            )
            if isinstance(relation.weight, ElemWiseParam) and mode.is_a(brainstate.mixin.Batching):
                fn_vmap = jax.vmap(fn_vmap)
                dg_weight = _sum_dim(_sum_dim(fn_vmap(df_hid), axis=-1), axis=0)
            else:
                dg_weight = _sum_dim(fn_vmap(df_hid))

            # update the weight gradients
            _update_dict(dG_weights, weight_path, dg_weight)  # update the weight gradients


def _init_param_dim_state(
    mode: brainstate.mixin.Mode,
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State],
    relation: HiddenParamOpRelation
):
    """
    Initialize the eligibility trace states for parameter dimensions.

    This function sets up the eligibility trace states for the weights and 
    differential functions (df) associated with a given relation. It assumes 
    that the batch size is the first dimension of the output shape if batching 
    is enabled.

    Args:
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.
        etrace_bwg (Dict[ETraceWG_Key, brainstate.State]): A dictionary to store the
            eligibility trace states, keyed by a unique identifier for each 
            weight group.
        relation (HiddenParamOpRelation): The relation object containing 
            information about the weights and hidden groups involved in the 
            computation.

    Raises:
        ValueError: If a relation with the same key has already been added to 
            the eligibility trace states.
    """
    # For the relation
    #
    #   h1, h2, ... = f(x, w)
    #
    # we need to initialize the eligibility trace states for the weight x and the df.

    # TODO: assume the batch size is the first dimension
    y_shape = relation.y.aval.shape
    batch_size = y_shape[0] if mode.has(brainstate.mixin.Batching) else None
    for group in relation.hidden_groups:
        group: HiddenGroup
        bwg_key = etrace_param_key(relation.path, relation.y, group.index)
        if bwg_key in etrace_bwg:  # The key should be unique
            raise ValueError(f'The relation {bwg_key} has been added. ')
        etrace_bwg[bwg_key] = EligibilityTrace(
            jax.tree.map(
                partial(_batched_zeros_like, batch_size, group.num_state),
                relation.weight.value
            )
        )


def _normalize_matrix_spectrum(matrix):
    # Compute the eigenvalues of the matrix
    eigenvalues = jnp.linalg.eigvals(matrix)

    # Get the maximum eigenvalue
    max_eigenvalue = jnp.max(jnp.abs(eigenvalues))

    # Normalize the matrix by dividing it by the maximum eigenvalue
    normalized_matrix = jax.lax.cond(
        max_eigenvalue > 1,
        lambda: matrix / max_eigenvalue,
        lambda: matrix,
    )

    return normalized_matrix


def _normalize_vector(v):
    max_elem = jnp.abs(v).max()
    normalized_vector = jax.lax.cond(
        max_elem > 1,
        lambda: v / max_elem,
        lambda: v,
    )

    # # Normalize the vector by dividing it by its norm
    # normalized_vector = v / jnp.linalg.norm(v)
    #
    return normalized_vector


def _update_param_dim_etrace_scan_fn(
    hist_etrace_vals: Dict[ETraceWG_Key, jax.Array],
    jacobians: Tuple[
        Dict[ETraceX_Key, jax.Array],  # the weight x
        Dict[ETraceDF_Key, jax.Array],  # the weight df
        Sequence[jax.Array],  # the hidden group Jacobians
    ],
    weight_path_to_vals: Dict[Path, PyTree],
    hidden_param_op_relations,
    mode: brainstate.mixin.Mode,
    normalize_matrix_spectrum: bool = False,
):
    """
    Update the eligibility trace values for parameter dimensions.

    This function updates the eligibility trace values for the parameter dimensions
    based on the provided Jacobians and the current mode. It computes the new eligibility
    trace values by applying vector-Jacobian products and incorporating the current
    Jacobian values.

    Args:
        hist_etrace_vals (Dict[ETraceWG_Key, jax.Array]): A dictionary containing
            historical eligibility trace values for the weight gradients, keyed by
            ETraceWG_Key.
        jacobians (Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array], Sequence[jax.Array]]):
            A tuple containing dictionaries of current Jacobian values for the weight x
            and df, and a sequence of hidden group Jacobians.
        weight_path_to_vals (Dict[Path, PyTree]): A dictionary mapping weight paths to
            their corresponding PyTree values.
        hidden_param_op_relations: A sequence of HiddenParamOpRelation objects representing
            the relationships between hidden parameters and operations.
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.

    Returns:
        Tuple[Dict[ETraceWG_Key, jax.Array], None]: A tuple containing a dictionary of
        updated eligibility trace values for the weight gradients, keyed by ETraceWG_Key,
        and None.
    """
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

    if normalize_matrix_spectrum:
        normalized_hid_group_jacobians = []
        for diag in hid_group_jacobians:
            fn = _normalize_matrix_spectrum
            for i in range(diag.ndim - 2):
                fn = jax.vmap(fn)
            normalized_hid_group_jacobians.append(fn(diag))
    else:
        normalized_hid_group_jacobians = hid_group_jacobians

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
        if isinstance(relation.weight, ElemWiseParam):
            x = None
        else:
            x = etrace_xs_at_t[id(relation.x)]

        def comp_dw_with_x(x_, df_):
            """
            Computes the vector-Jacobian product (VJP) of the output with respect to the weight parameter.

            Args:
                x_: The input to the weight operation (can be None for element-wise parameters).
                df_: The cotangent (adjoint) vector for the output, used in the VJP computation.

            Returns:
                The VJP result, representing the gradient of the output with respect to the weight,
                contracted with the provided cotangent vector.
            """

            def to_y(w):
                # Returns the mantissa (unitless value) of the output of the weight operation.
                return u.get_mantissa(etrace_op.xw_to_y(x_, w))

            # Compute the VJP of to_y with respect to weight_val, evaluated at df_.
            return jax.vjp(to_y, weight_val)[1](df_)[0]

        @partial(jax.vmap, in_axes=-1, out_axes=-1)
        def comp_dw_without_x(df_):
            """
            Vectorized version of fn_dw for cases where x is not None.

            If batching is enabled, applies fn_dw over the batch dimension using jax.vmap.
            Otherwise, applies fn_dw directly.

            Args:
                df_: The cotangent (adjoint) vector for the output, used in the VJP computation.

            Returns:
                The VJP result(s) for the provided df_.
            """
            if mode.has(brainstate.mixin.Batching):
                return jax.vmap(comp_dw_with_x)(x, df_)
            else:
                return comp_dw_with_x(x, df_)

        for group in relation.hidden_groups:
            group: HiddenGroup

            #
            # Step 2:
            #
            # compute the current step weight gradients:
            #
            #       \partial h^t / \partial W^t = vjp(f(x, w))(df)
            #
            df = etrace_ys_at_t[etrace_df_key(relation.y, group.index)]
            # jax.debug.print('df = {g}', g=jax.tree.map(lambda x: jnp.abs(x).max(), df))

            #
            # vmap over the different hidden states,
            #
            # x: (n_input, ..., )
            # df: (n_hidden, ..., n_state)
            # phg_to_pw: (n_param, ..., n_state)
            phg_to_pw = comp_dw_without_x(df)
            phg_to_pw = jax.tree.map(_normalize_vector, phg_to_pw)
            # jax.debug.print('phg_to_pw = {g}', g=jax.tree.map(lambda x: jnp.abs(x).max(), phg_to_pw))

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
            diag = normalized_hid_group_jacobians[group.index]

            #
            # vmap over j, over the different hidden states \partial h_i^t / \partial h_j^t
            #
            # d: (n_hidden, ..., [n_state])
            # old_bwg: (n_param, ..., [n_state])
            old_bwg = hist_etrace_vals[w_key]
            fn_bwg_pre = lambda d: _sum_dim(
                jax.vmap(etrace_op.yw_to_w, in_axes=-1, out_axes=-1)(d, old_bwg), axis=-1
            )
            if isinstance(relation.weight, ElemWiseParam) and mode.is_a(brainstate.mixin.Batching):
                raise NotImplementedError

            #
            # vmap over i, over the different hidden states \partial h_i^t / \partial h_j^t
            #
            # diag: (n_hidden, ..., [n_state], n_state)
            # old_bwg: (n_param, ..., n_state)
            # new_bwg_pre: (n_param, ..., n_state)
            new_bwg_pre = jax.vmap(fn_bwg_pre, in_axes=-2, out_axes=-1)(diag)

            #
            # Step 4:
            #
            # update: eligibility trace * hidden diagonal Jacobian + new hidden df
            #        ϵ^t = ϵ^t_{pre} + df^t, where D_h is the hidden-to-hidden Jacobian diagonal matrix.
            #
            new_bwg = jax.tree.map(u.math.add, new_bwg_pre, phg_to_pw, is_leaf=u.math.is_quantity)
            new_bwg = jax.tree.map(_normalize_vector, new_bwg)
            new_etrace_bwg[w_key] = new_bwg

    return new_etrace_bwg, None


def _solve_param_dim_weight_gradients(
    hist_etrace_data: Dict[ETraceWG_Key, PyTree],  # the history etrace data
    dG_weights: Dict[Path, dG_Weight],  # weight gradients
    dG_hidden_groups: Sequence[jax.Array],  # hidden group gradients
    weight_hidden_relations: Sequence[HiddenParamOpRelation],
    mode: brainstate.mixin.Mode,
):
    """
    Compute and update the weight gradients for parameter dimensions using eligibility trace data.

    This function calculates the weight gradients by utilizing the eligibility trace data and the 
    hidden-to-hidden Jacobians. It applies a correction factor to avoid exponential smoothing bias 
    at the beginning of the computation.

    Args:
        hist_etrace_data (Dict[ETraceWG_Key, PyTree]): A dictionary containing historical eligibility 
            trace data for the weight gradients, keyed by ETraceWG_Key.
        dG_weights (Dict[Path, dG_Weight]): A dictionary to store the computed weight gradients, 
            keyed by the path of the weight.
        dG_hidden_groups (Sequence[jax.Array]): A sequence of hidden group gradients, with the same 
            length as the total number of hidden groups.
        weight_hidden_relations (Sequence[HiddenParamOpRelation]): A sequence of HiddenParamOpRelation 
            objects representing the relationships between hidden parameters and operations.
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.

    Returns:
        None: The function updates the dG_weights dictionary in place with the computed weight gradients.
    """
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
        yw_to_w = jax.vmap(etrace_op.yw_to_w) if mode.has(brainstate.mixin.Batching) else etrace_op.yw_to_w

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
            # dimensionless processing
            etrace_data, fn_unit_restore = _remove_units(etrace_data)
            dg_hidden, _ = _remove_units(dg_hidden)

            #
            # etrace_data: [n_batch, n_param, ..., n_state]
            #               or,
            #              [n_param, ..., n_state]
            # dg_hidden:   [n_batch, n_hidden, ..., n_state]
            #               or,
            #              [n_hidden, ..., n_state]
            dg_weight = _sum_dim(
                jax.vmap(yw_to_w, in_axes=-1, out_axes=-1)(dg_hidden, etrace_data)
            )
            # unit restoration
            dg_weight = fn_unit_restore(dg_weight)

            # update the weight gradients
            _update_dict(temp_data, weight_path, dg_weight)

    #
    # Step 3:
    #
    # sum up the batched weight gradients
    if mode.has(brainstate.mixin.Batching):
        for key, val in temp_data.items():
            temp_data[key] = jax.tree.map(lambda x: u.math.sum(x, axis=0), val)

    # update the weight gradients
    for key, val in temp_data.items():
        _update_dict(dG_weights, key, val)


def _remove_units(xs_maybe_quantity: brainstate.typing.PyTree):
    """
    Removes units from a PyTree of quantities, returning a unitless PyTree and a function to restore the units.

    This function traverses a PyTree structure, removing units from each quantity and returning a new PyTree
    with the same structure but without units. It also returns a function that can be used to restore the
    original units to the unitless PyTree.

    Args:
        xs_maybe_quantity (brainstate.typing.PyTree): A PyTree structure containing quantities with units.

    Returns:
        Tuple[brainstate.typing.PyTree, Callable]: A tuple containing:
            - A PyTree with the same structure as the input, but with units removed from each quantity.
            - A function that takes a unitless PyTree and restores the original units to it.
    """
    leaves, treedef = jax.tree.flatten(xs_maybe_quantity, is_leaf=u.math.is_quantity)
    new_leaves, units = [], []
    for leaf in leaves:
        leaf, unit = u.split_mantissa_unit(leaf)
        new_leaves.append(leaf)
        units.append(unit)

    def restore_units(xs_unitless: brainstate.typing.PyTree):
        leaves, treedef2 = jax.tree.flatten(xs_unitless)
        assert treedef == treedef2, 'The tree structure should be the same. '
        new_leaves = [
            leaf if unit.dim.is_dimensionless else leaf * unit
            for leaf, unit in zip(leaves, units)
        ]
        return jax.tree.unflatten(treedef, new_leaves)

    return jax.tree.unflatten(treedef, new_leaves), restore_units


def _sum_dim(xs: jax.Array, axis: int = -1):
    """
    Sums the elements along the last dimension of each array in a PyTree.

    This function applies a sum operation along the last dimension of each array
    within a PyTree structure. It is useful for reducing the dimensionality of
    arrays by aggregating values along the specified axis.

    Args:
        xs (jax.Array): A PyTree of arrays where each array will have its last
                        dimension summed.

    Returns:
        jax.Array: A PyTree with the same structure as the input, where each array
                   has been reduced by summing over its last dimension.
    """
    return jax.tree.map(lambda x: u.math.sum(x, axis=axis), xs)


def _zeros_like_batch_or_not(
    batch_size: Optional[int],
    x: jax.Array
):
    """
    Create a zeros array with the same shape and type as the input array, 
    optionally including a batch dimension.

    This function generates a zeros array that matches the shape and data type 
    of the input array `x`. If a batch size is provided, the zeros array will 
    include an additional batch dimension at the beginning.

    Args:
        batch_size (Optional[int]): The size of the batch. If provided, the 
            zeros array will include a batch dimension. If None, the zeros 
            array will have the same shape as `x`.
        x (jax.Array): The input array whose shape and data type are used as 
            a reference for creating the zeros array.

    Returns:
        jax.Array: A zeros array with the same shape and data type as the 
        input array, optionally including a batch dimension if `batch_size` 
        is provided.
    """
    if batch_size is not None:
        assert isinstance(batch_size, int), 'The batch size should be an integer. '
        return u.math.zeros((batch_size,) + x.shape[1:], x.dtype)
    else:
        return u.math.zeros_like(x)


def _reset_state_in_a_dict(
    state_dict: Dict[Any, brainstate.State],
    batch_size: Optional[int],
):
    """
    Reset the values in a dictionary of states to zero.

    This function iterates over a dictionary of states and resets each state's 
    value to a zero array. The shape of the zero array is determined by the 
    original shape of the state's value and the specified batch size.

    Args:
        state_dict (Dict[Any, brainstate.State]): A dictionary where keys are any
            type and values are brainstate.State objects. Each state's value will be
            reset to a zero array.
        batch_size (Optional[int]): The size of the batch. If provided, the 
            zero array will include a batch dimension; otherwise, it will not.

    Returns:
        None: The function modifies the state_dict in place, resetting each 
        state's value to a zero array.
    """
    for k, v in state_dict.items():
        state_dict[k].value = jax.tree.map(partial(_zeros_like_batch_or_not, batch_size), v.value)


def _numel(pytree: PyTree):
    """
    Calculate the total number of elements in a PyTree.

    This function traverses a PyTree structure and sums up the number of elements
    in each array contained within the PyTree.

    Args:
        pytree (PyTree): A PyTree structure, which can be a nested combination of
                         lists, tuples, and dictionaries containing JAX arrays.

    Returns:
        int: The total number of elements across all arrays in the PyTree.
    """
    return sum(u.math.size(x) for x in jax.tree_leaves(pytree))


def _is_weight_need_full_grad(
    relation: HiddenParamOpRelation,
    mode: brainstate.mixin.Mode
):
    """
    Determine whether the weight requires a full gradient computation.

    This function evaluates the type of gradient computation needed for a given weight
    based on its characteristics and the current mode. It decides whether to use an
    O(n^2) algorithm for full gradient computation or an O(n) algorithm for approximate
    gradient computation.

    Args:
        relation (HiddenParamOpRelation): The relation object containing information
            about the weights and hidden groups involved in the computation.
        mode (brainstate.mixin.Mode): The mode indicating whether batching is enabled.

    Returns:
        bool: True if the weight requires a full gradient computation using the O(n^2)
        algorithm, False if an approximate gradient computation using the O(n) algorithm
        is sufficient.
    """
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

    batch_size = relation.x.aval.shape[0] if mode.has(brainstate.mixin.Batching) else 1
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
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.

    """

    __module__ = 'brainscale'
    graph_executor: ETraceVjpGraphExecutor

    def __init__(
        self,
        model: brainstate.nn.Module,
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
        graph_executor = ETraceVjpGraphExecutor(model, vjp_method=vjp_method)

        # super initialization
        super().__init__(model=model, name=name, graph_executor=graph_executor)

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
        return out

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
        input_is_multi_step = has_multistep_data(*args)

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, oth_state_vals, write=False)

        # necessary jacobian information of the weights
        (
            out,
            hidden_vals,
            oth_state_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps
        ) = self.graph_executor.solve_h2w_h2h_jacobian(*args)

        # eligibility trace update
        #
        # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        etrace_vals = self._update_etrace_data(
            running_index,
            etrace_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            weight_vals,
            input_is_multi_step,
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
        input_is_multi_step = has_multistep_data(*args)

        # state value assignment
        assign_state_values_v2(self.param_states, weight_vals, write=False)
        assign_state_values_v2(self.hidden_states, hidden_vals, write=False)
        assign_state_values_v2(self.other_states, othstate_vals, write=False)

        # necessary gradients of the weights
        (
            out,
            hiddens,
            oth_states,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            residuals
        ) = self.graph_executor.solve_h2w_h2h_l2h_jacobian(*args)

        # eligibility trace update
        #
        # "self._update_etrace_data()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        new_etrace_vals = self._update_etrace_data(
            running_index,
            etrace_vals,
            hid2weight_jac_single_or_multi_steps,
            hid2hid_jac_single_or_multi_steps,
            weight_vals,
            input_is_multi_step
        )

        # returns
        old_etrace_vals = etrace_vals
        fwd_out = (out, hiddens, oth_states, new_etrace_vals)
        fwd_res = (
            residuals,
            (
                old_etrace_vals
                if self.graph_executor.is_multi_step_vjp else
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
            dg_hid_perturb_or_dl2h
        ) = jax.tree.unflatten(in_tree, cts_out)

        #
        # get the gradients of the hidden states at the last time step
        #
        if self.graph_executor.is_single_step_vjp:
            # TODO: the correspondence between the hidden states and the gradients
            #       should be checked.
            #
            assert len(dg_etrace_params) == 0  # gradients all etrace weights are updated by the RTRL algorithm
            assert len(self.graph.hidden_perturb.perturb_vars) == len(dg_hid_perturb_or_dl2h)
            dl2h_at_t_or_t_minus_1 = self.graph.hidden_perturb.perturb_data_to_hidden_group_data(
                dg_hid_perturb_or_dl2h,
                self.graph.hidden_groups,
            )

        else:
            assert len(dg_last_hiddens) == len(self.hidden_states)
            assert set(dg_last_hiddens.keys()) == set(self.hidden_states.keys()), (
                f'The hidden states should be the same. Bug got \n'
                f'{set(dg_last_hiddens.keys())}\n'
                f'!=\n'
                f'{set(self.hidden_states.keys())}'
            )
            dl2h_at_t_or_t_minus_1 = [
                group.concat_hidden(
                    [
                        # dimensionless processing
                        u.get_mantissa(dg_last_hiddens[path])
                        for path in group.hidden_paths
                    ]
                )
                for group in self.graph.hidden_groups
            ]

        #
        # [4] Compute the gradients of the weights
        #
        # the gradients of the weights are computed through the RTRL algorithm.
        #
        # "self._solve_weight_gradients()" is a protocol method that should be implemented in the subclass.
        # It's logic may be different for different etrace algorithms.
        #
        dg_weights = self._solve_weight_gradients(
            running_index,
            etrace_vals_at_t_or_t_minus_1,
            dl2h_at_t_or_t_minus_1,
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
        dl_to_hidden_groups: Sequence[jax.Array],
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
          dl_to_hidden_groups: Dict[HiddenOutVar, jax.Array], the gradients of the loss-to-hidden
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
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: Sequence[jax.Array],
        weight_vals: WeightVals,
        input_is_multi_step: bool,
    ) -> ETraceVals:
        """
        The method to update the eligibility trace data.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
          running_index: Optional[int], the running index.
          etrace_vals_util_t_1: ETraceVals, the history eligibility trace data that have accumulated util :math:`t-1`.
          hid2weight_jac_single_or_multi_times: ETraceVals, the current eligibility trace data at the time :math:`t`.
          hid2hid_jac_single_or_multi_times: The data for computing the hidden-to-hidden Jacobian at the time :math:`t`.
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

    Parameters
    -----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    vjp_method: str, optional
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.

    decay_or_rank: float, int
        The exponential smoothing factor for the eligibility trace.
        If it is a float, it is the decay factor, should be in the range of (0, 1).
        If it is an integer, it is the number of approximation rank for the algorithm, should be greater than 0.
    name: str, optional
        The name of the etrace algorithm.
    mode: brainscale.mixin.Mode
        The computing mode, indicating the batching information.
    """

    __module__ = 'brainscale'

    # the spatial gradients of the weights
    etrace_xs: Dict[ETraceX_Key, brainstate.State]

    # the spatial gradients of the hidden states
    etrace_dfs: Dict[ETraceDF_Key, brainstate.State]

    # the mapping from the etrace x to the weight operations
    etrace_xs_to_weights = Dict[ETraceX_Key, List[Path]]

    # the exponential smoothing decay factor
    decay: float

    def __init__(
        self,
        model: brainstate.nn.Module,
        decay_or_rank: float | int,
        mode: Optional[brainstate.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

        # computing mode
        if mode is None:
            self.mode = brainstate.environ.get('mode', brainstate.mixin.Mode())
        else:
            self.mode = mode
        assert isinstance(self.mode, brainstate.mixin.Mode), 'The mode should be an instance of brainstate.mixin.Mode.'

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
        for relation in self.graph.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            _init_IO_dim_state(
                self.etrace_xs,
                self.etrace_dfs,
                self.etrace_xs_to_weights,
                self.graph_executor.state_id_to_path,
                relation,
                self.mode,
            )

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_xs, batch_size)
        _reset_state_in_a_dict(self.etrace_dfs, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Tuple[Dict, Dict]:
        """
        Get the eligibility trace of the given weight.

        The eligibility trace contains the following structures:

        """
        self._assert_compiled()

        # the weight ID
        weight_id = (
            id(weight)
            if isinstance(weight, brainstate.ParamState) else
            id(self.graph_executor.path_to_states[weight])
        )

        etrace_xs = dict()
        etrace_dfs = dict()
        find_this_weight = False
        for relation in self.graph.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            if id(relation.weight) != weight_id:
                continue
            find_this_weight = True

            # get the weight_op input
            wx_var = etrace_x_key(relation.x)
            if wx_var is not None:
                etrace_xs[wx_var] = self.etrace_xs[wx_var].value

            # get the weight_op df
            wy_var = relation.y
            for group in relation.hidden_groups:
                group: HiddenGroup
                df_key = etrace_df_key(wy_var, group.index)
                etrace_dfs[df_key] = self.etrace_dfs[df_key].value
        if not find_this_weight:
            raise ValueError(f'Do not the etrace of the given weight: {weight}.')
        return etrace_xs, etrace_dfs

    def _get_etrace_data(self) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        """
        Get the eligibility trace data at the last time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Returns:
            ETraceVals, the eligibility trace data.
        """
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
        """Assign the eligibility trace data to the states at the current time-step.

        .. note::

            This is the protocol method that should be implemented in the subclass.

        Args:
            hist_etrace_vals: ETraceVals, the eligibility trace data.
        """
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
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: HiddenGroupJacobian,
        weight_vals: WeightVals,
        input_is_multi_step: bool,
    ) -> Tuple[
        Dict[ETraceX_Key, jax.Array],
        Dict[ETraceDF_Key, jax.Array]
    ]:
        """Update the eligibility trace data for a given timestep.

        This method implements the core update equations for the eligibility trace
        algorithm with input-output dimensional complexity. It processes historical
        trace values along with current Jacobians to compute the updated eligibility
        traces according to the algorithm's update rules.

        Args:
            running_index: Optional[int]
                The current timestep index. Used for decay correction factors.
            hist_etrace_vals: Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]
                The eligibility trace values from the previous timestep, containing:
                - Dictionary mapping weight inputs to their trace values
                - Dictionary mapping differential functions to their trace values
            hid2weight_jac_single_or_multi_times: Hid2WeightJacobian
                The current hidden-to-weight Jacobians at time t (or t-1 depending on vjp_method).
            hid2hid_jac_single_or_multi_times: HiddenGroupJacobian
                The current hidden-to-hidden Jacobians for propagating gradients.
            weight_vals: WeightVals
                The current values of the model weights.

        Returns:
            Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]:
                Updated eligibility trace values for both input traces and differential
                function traces, computed according to the exponential smoothing rules
                of the algorithm.
        """
        #
        # "running_index":
        #            the running index
        #
        # "hist_etrace_vals":
        #            the history etrace values,
        #            including the x and df values, see "etrace_xs" and "etrace_dfs".
        #
        # "hid2weight_jac_single_or_multi_times":
        #           the current etrace values at the time "t", \epsilon^t, if vjp_time == "t".
        #           Otherwise, the etrace values at the time "t-1", \epsilon^{t-1}.
        #
        # "hid2hid_jac_single_or_multi_times":
        #           the data for computing the hidden-to-hidden Jacobian at the time "t".
        #
        # "weight_path_to_vals":
        #           the weight values.
        #

        scan_fn = partial(
            _update_IO_dim_etrace_scan_fn,
            hid_weight_op_relations=self.graph.hidden_param_op_relations,
            decay=self.decay,
        )

        if input_is_multi_step:
            hist_etrace_vals = jax.lax.scan(
                scan_fn,
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        else:
            hist_etrace_vals = scan_fn(
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        return hist_etrace_vals

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Tuple[
            Dict[ETraceX_Key, jax.Array],
            Dict[ETraceDF_Key, jax.Array]
        ],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """Compute weight gradients using eligibility trace data and loss gradients.

        This method implements the final stage of the eligibility trace algorithm, where
        the eligibility traces are combined with the loss gradients to compute the weight
        parameter gradients. It follows the mathematical equation:

        ∇_θ L = ∑ (∂L/∂h) ⊙ ϵ

        where ϵ represents the eligibility traces and ∂L/∂h are the gradients of
        the loss with respect to hidden states.

        Args:
            running_index: int
                The current timestep index, used for correction factor calculation.
            etrace_h2w_at_t: Tuple[Dict[ETraceX_Key, jax.Array], Dict[ETraceDF_Key, jax.Array]]
                The eligibility trace data at the current timestep, containing:
                - Dictionary mapping weight inputs to their trace values
                - Dictionary mapping differential functions to their trace values
            dl_to_hidden_groups: Sequence[jax.Array]
                Gradients of the loss with respect to each hidden group/state.
            weight_vals: Dict[Path, PyTree]
                Current values of the model weights.
            dl_to_nonetws_at_t: Dict[Path, PyTree]
                Gradients for non-eligibility trace weights computed through standard backprop.
            dl_to_etws_at_t: Optional[Dict[Path, PyTree]]
                Optional additional gradients for eligibility trace weights.

        Returns:
            Dict[Path, jax.Array]: Computed gradients for all weights in the model.
        """

        #
        # dl_to_hidden_groups:
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
            dl_to_hidden_groups,
            self.graph.hidden_param_op_relations,
            weight_vals,
            running_index,
            self.decay,
            self.mode,
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

    Parameters
    -----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    vjp_method: str, optional
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.
    name: str, optional
        The name of the etrace algorithm.
    mode: brainscale.mixin.Mode
        The computing mode, indicating the batching behavior.
    """

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State]

    def __init__(
        self,
        model: brainstate.nn.Module,
        mode: Optional[brainstate.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

        # computing mode
        if mode is None:
            self.mode = brainstate.environ.get('mode', brainstate.mixin.Mode())
        else:
            self.mode = mode
        assert isinstance(self.mode, brainstate.mixin.Mode), (
            f'The mode should be an instance of brainstate.mixin.Mode. But we got {self.mode}.'
        )

    def init_etrace_state(self, *args, **kwargs):
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See
        `.compile_graph()` for the details.
        """
        # The states of batched weight gradients
        self.etrace_bwg = dict()
        for relation in self.graph.hidden_param_op_relations:
            _init_param_dim_state(self.mode, self.etrace_bwg, relation)

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Dict:
        """
        Get the eligibility trace of the given weight.

        The eligibility trace contains the following structures:

        """

        self._assert_compiled()

        # get the wight id
        weight_id = (
            id(weight)
            if isinstance(weight, brainstate.ParamState) else
            id(self.graph_executor.path_to_states[weight])
        )

        find_this_weight = False
        etraces = dict()
        for relation in self.graph.hidden_param_op_relations:
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
        """Retrieve the current eligibility trace data from all trace states.

        This method collects all eligibility trace values from the internal state dictionary,
        extracting the current values from the brainstate.State objects that store them.
        It returns these values in a dictionary with the same keys as the original state
        dictionary, making the current trace values available for processing.

        This is an internal method used in the parameter dimension eligibility trace algorithm
        to access the current trace state for updates and gradient calculations.

        Returns:
            Dict[ETraceWG_Key, jax.Array]: A dictionary mapping eligibility trace keys to
                their current values. Each key represents a specific trace component
                (typically involving a parameter and hidden state relationship), and
                the corresponding value represents the accumulated eligibility trace.
        """
        return {
            k: v.value
            for k, v in self.etrace_bwg.items()
        }

    def _assign_etrace_data(self, etrace_vals: Dict) -> None:
        """Assign eligibility trace values to their corresponding state objects.

        This method updates the internal eligibility trace state dictionary (etrace_bwg)
        with new values from the provided dictionary. It iterates through each key-value
        pair in the input dictionary and assigns the value to the corresponding state
        object's value attribute.

        This is an implementation of the abstract method from the parent class,
        customized for the parameter dimension eligibility trace algorithm which
        stores traces in a single dictionary rather than separate ones for inputs
        and differential functions.

        Args:
            etrace_vals: Dict[ETraceWG_Key, jax.Array]
                Dictionary mapping eligibility trace keys to their updated values.
                Each key represents a specific parameter-hidden state relationship,
                and the value represents the updated eligibility trace value.

        Returns:
            None
        """
        for x, val in etrace_vals.items():
            self.etrace_bwg[x].value = val

    def _update_etrace_data(
        self,
        running_index: Optional[int],
        hist_etrace_vals: Dict[ETraceWG_Key, PyTree],
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: HiddenGroupJacobian,
        weight_vals: Dict[Path, PyTree],
        input_is_multi_step: bool,
    ) -> Dict[ETraceWG_Key, PyTree]:
        """Update eligibility trace data for the parameter dimension-based algorithm.

        This method implements the core update equation for the D-RTRL algorithm's eligibility traces:

        ε^t ≈ D^t·ε^{t-1} + diag(D_f^t)⊗x^t

        It uses JAX's scan operation to efficiently process the historical trace values and
        combines them with current Jacobians to compute updated traces according to the
        parameter-dimension approximation approach.

        Args:
            running_index: Optional[int]
                Current timestep counter, used for correcting exponential smoothing bias.
            hist_etrace_vals: Dict[ETraceWG_Key, PyTree]
                Dictionary containing historical eligibility trace values from previous timestep.
                Keys are tuples identifying parameter-hidden state relationships.
            hid2weight_jac_single_or_multi_times: Hid2WeightJacobian
                Jacobians of hidden states with respect to weights at the current timestep.
                Contains input gradients and differential function gradients.
            hid2hid_jac_single_or_multi_times: HiddenGroupJacobian
                Jacobians between hidden states (recurrent connections) at the current timestep.
            weight_vals: Dict[Path, PyTree]
                Dictionary mapping paths to current weight values in the model.

        Returns:
            Dict[ETraceWG_Key, PyTree]: Updated eligibility trace values dictionary with the
                same structure as hist_etrace_vals but containing new values for the current timestep.
        """

        scan_fn = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=self.graph.hidden_param_op_relations,
            mode=self.mode,
        )

        if input_is_multi_step:
            new_etrace = jax.lax.scan(
                scan_fn,
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                )
            )[0]

        else:
            new_etrace = scan_fn(
                hist_etrace_vals,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                )
            )[0]

        return new_etrace

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree],
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[WeightID, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """Compute weight gradients using parameter dimension eligibility traces.

        This method implements the parameter dimension D-RTRL algorithm's weight gradient
        computation. It combines the eligibility traces with the gradients of the loss
        with respect to hidden states to compute the full parameter gradients according to:

        ∇_θ L = ∑_{t' ∈ T} ∂L^{t'}/∂h^{t'} ∘ ε^{t'}

        Where ε represents the eligibility traces and ∂L/∂h are the gradients of the loss
        with respect to hidden states.

        Args:
            running_index: int
                Current timestep counter used for bias correction.
            etrace_h2w_at_t: Dict[ETraceWG_Key, PyTree]
                Eligibility trace values at the current timestep, mapping parameter-hidden
                state relationship keys to trace values.
            dl_to_hidden_groups: Sequence[jax.Array]
                Gradients of the loss with respect to hidden states at the current timestep.
            weight_vals: Dict[WeightID, PyTree]
                Current values of all weights in the model.
            dl_to_nonetws_at_t: Dict[Path, PyTree]
                Gradients of non-eligibility trace parameters at the current timestep.
            dl_to_etws_at_t: Optional[Dict[Path, PyTree]]
                Optional additional gradients for eligibility trace parameters at the
                current timestep.

        Returns:
            Dict[Path, PyTree]: Dictionary mapping parameter paths to their gradient values.
        """
        dG_weights = {path: None for path in self.param_states}

        # update the etrace weight gradients
        _solve_param_dim_weight_gradients(
            etrace_h2w_at_t,
            dG_weights,
            dl_to_hidden_groups,
            self.graph.hidden_param_op_relations,
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

    Parameters
    -----------
    model: Callable
        The model function, which receives the input arguments and returns the model output.
    vjp_method: str, optional
        The method for computing the VJP. It should be either "single-step" or "multi-step".

        - "single-step": The VJP is computed at the current time step, i.e., $\partial L^t/\partial h^t$.
        - "multi-step": The VJP is computed at multiple time steps, i.e., $\partial L^t/\partial h^{t-k}$,
          where $k$ is determined by the data input.
    name: str, optional
        The name of the etrace algorithm.
    decay_or_rank: float, int
        The exponential smoothing factor for the eligibility trace. If it is a float,
        it is the decay factor, should be in the range of (0, 1). If it is an integer,
        it is the number of approximation rank for the algorithm, should be greater than 0.
    mode: brainscale.mixin.Mode
        The computing mode, indicating the batching behavior.
    """

    # the spatial gradients of the weights
    etrace_xs: Dict[ETraceX_Key, brainstate.State]

    # the spatial gradients of the hidden states
    etrace_dfs: Dict[ETraceDF_Key, brainstate.State]

    # the mapping from the etrace x to the weight operations
    etrace_xs_to_weights = Dict[ETraceX_Key, List[Path]]

    # batch of weight gradients
    etrace_bwg: Dict[ETraceWG_Key, brainstate.State]

    # the exponential smoothing decay factor
    decay: float

    def __init__(
        self,
        model: brainstate.nn.Module,
        decay_or_rank: float | int,
        mode: Optional[brainstate.mixin.Mode] = None,
        name: Optional[str] = None,
        vjp_method: str = 'single-step'
    ):
        super().__init__(model, name=name, vjp_method=vjp_method)

        # computing mode
        if mode is None:
            self.mode = brainstate.environ.get('mode', brainstate.mixin.Mode())
        else:
            self.mode = mode
        assert isinstance(self.mode, brainstate.mixin.Mode), 'The mode should be an instance of brainstate.mixin.Mode.'

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

        for relation in self.graph.hidden_param_op_relations:
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
                    self.graph_executor.state_id_to_path,
                    relation,
                    self.mode,
                )

    def reset_state(self, batch_size: int = None, **kwargs):
        """
        Reset the eligibility trace states.

        This function resets the internal state of the eligibility traces, which are used
        in the computation of gradients in the etrace algorithm. It is typically called
        at the beginning of a new batch or sequence to ensure that the state is clean.

        Parameters
        -----------
        batch_size : int, optional
            The size of the batch for which the state is being reset. If not provided,
            the default behavior is to reset the state without considering batch size.
        
        **kwargs
            Additional keyword arguments that may be used for resetting the state.
            These are not explicitly used in this function but can be passed for
            compatibility with other functions or methods that require them.

        Returns:
        --------
        None
            This function does not return any value. It performs an in-place reset
            of the internal state variables.
        """
        self.running_index.value = 0
        _reset_state_in_a_dict(self.etrace_xs, batch_size)
        _reset_state_in_a_dict(self.etrace_dfs, batch_size)
        _reset_state_in_a_dict(self.etrace_bwg, batch_size)

    def get_etrace_of(self, weight: brainstate.ParamState | Path) -> Tuple[Dict, Dict, Dict]:
        """
        Retrieve the eligibility trace for a specified weight.

        This function extracts the eligibility trace data associated with a given weight,
        which includes the spatial gradients of the weight inputs, the spatial gradients
        of the hidden states, and the batched weight gradients.

        Parameters
        -----------
        weight : brainstate.ParamState | Path
            The weight for which the eligibility trace is to be retrieved. It can be
            specified either as a `brainstate.ParamState` object or a `Path` object.

        Returns:
        --------
        Tuple[Dict, Dict, Dict]
            A tuple containing three dictionaries:
            - etrace_xs: The spatial gradients of the weight inputs.
            - etrace_dfs: The spatial gradients of the hidden states.
            - etrace_bws: The batched weight gradients.

        Raises:
        -------
        ValueError
            If the eligibility trace for the specified weight cannot be found.
        """

        self._assert_compiled()

        # the weight ID
        weight_id = (
            id(weight)
            if isinstance(weight, brainstate.ParamState) else
            id(self.graph_executor.path_to_states[weight])
        )

        etrace_xs = dict()
        etrace_dfs = dict()
        etrace_bws = dict()
        find_this_weight = False
        for relation in self.graph.hidden_param_op_relations:
            relation: HiddenParamOpRelation
            if id(relation.weight) != weight_id:
                continue
            find_this_weight = True

            wx_var = etrace_x_key(relation.x)
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
        """Retrieve all eligibility trace data from internal state dictionaries.

        This method collects the current eligibility trace values from all three internal
        state dictionaries that store different components of the trace information:
        - etrace_xs: Input spatial gradients
        - etrace_dfs: Hidden state differential function values
        - etrace_wgrads: Weight gradients in parameter dimension

        It extracts the current values from the brainstate.State objects and returns them
        as a tuple of dictionaries with the same structure as the state dictionaries.

        This method is used internally during the update process to provide the current
        trace state for computation in the hybrid dimension algorithm.

        Returns:
            Tuple[Dict, ...]: A tuple containing three dictionaries:
                - Input spatial gradients (etrace_xs)
                - Hidden state differential values (etrace_dfs)
                - Weight gradients (etrace_wgrads)
        """
        etrace_xs = {x: val.value for x, val in self.etrace_xs.items()}
        etrace_dfs = {x: val.value for x, val in self.etrace_dfs.items()}
        etrace_wgrads = {x: val.value for x, val in self.etrace_bwg.items()}
        return etrace_xs, etrace_dfs, etrace_wgrads

    def _assign_etrace_data(self, etrace_vals: Sequence[Dict]) -> None:
        """Assign eligibility trace values to their corresponding state objects.

        This method updates the eligibility trace states with new values provided in the input
        dictionary. For the parameter dimension algorithm, it iterates through each key-value
        pair in the input dictionary and assigns the value to the corresponding state's value
        attribute in the etrace_bwg dictionary.

        This is an implementation of the abstract method from the parent ETraceVjpAlgorithm class,
        customized for storing traces specific to this algorithm's approach.

        Args:
            etrace_vals: Sequence[Dict]
                Dictionary mapping eligibility trace keys to their updated values.
                Each key identifies a specific weight-hidden state relationship,
                and the value contains the updated trace information.
        """
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
        hid2weight_jac_single_or_multi_times: Hid2WeightJacobian,
        hid2hid_jac_single_or_multi_times: Hid2HidJacobian,
        weight_vals: Dict[Path, PyTree],
        input_is_multi_step: bool,
    ) -> Tuple[Dict, ...]:
        """
        Update eligibility trace data for the hybrid dimension algorithm.

        This method combines the approaches from both IO dimension and parameter dimension
        algorithms to update eligibility traces. It decides which algorithm to use for each
        weight-hidden relationship based on the complexity characteristics of each operation.

        The hybrid algorithm chooses between:
        - IO dimension approach (O(BI+BO) complexity) when I+O < theta
        - Parameter dimension approach (O(B*theta) complexity) when I+O >= theta

        Where B is batch size, I is input dimensions, O is output dimensions, and theta is
        the number of parameters.

        Args:
            running_index: Optional[int]
                Current timestep counter used for decay corrections.
            hist_etrace_vals: Tuple[Dict, ...]
                Historical eligibility trace values as a tuple containing three dictionaries:
                (etrace_xs, etrace_dfs, etrace_wgrads) for input traces, differential function
                traces, and weight gradient traces respectively.
            hid2weight_jac_single_or_multi_times: Hid2WeightJacobian
                Jacobians of hidden states with respect to weights at current timestep.
            hid2hid_jac_single_or_multi_times: Hid2HidJacobian
                Jacobians of hidden states with respect to previous hidden states.
            weight_vals: Dict[Path, PyTree]
                Current values of all weights in the model.

        Returns:
            Tuple[Dict, ...]: Updated eligibility trace values as a tuple of three dictionaries
                containing the updated traces for inputs, differential functions, and weight
                gradients, maintaining the same structure as the input hist_etrace_vals.
        """

        # the history etrace values
        hist_xs, hist_dfs, hist_bwg = hist_etrace_vals

        # ---- separate the etrace gradients into two parts --- #
        #
        #  1. O(n^2) etrace gradients
        #  2. O(n) etrace gradients
        #

        on_weight_hidden_relations = []
        on2_weight_hidden_relations = []
        for relation in self.graph.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                on2_weight_hidden_relations.append(relation)
            else:
                on_weight_hidden_relations.append(relation)

        scan_fn_on2 = partial(
            _update_param_dim_etrace_scan_fn,
            weight_path_to_vals=weight_vals,
            hidden_param_op_relations=on2_weight_hidden_relations,
            mode=self.mode,
        )
        scan_fn_on = partial(
            _update_IO_dim_etrace_scan_fn,
            hid_weight_op_relations=self.graph.hidden_param_op_relations,
            decay=self.decay,
        )

        if input_is_multi_step:
            # ---- O(n^2) etrace gradients update ---- #
            new_bwg = jax.lax.scan(
                scan_fn_on2,
                hist_bwg,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                )
            )[0]

            # ---- O(n) etrace gradients update ---- #
            new_xs, new_dfs = jax.lax.scan(
                scan_fn_on,
                (hist_xs, hist_dfs),
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        else:
            # ---- O(n^2) etrace gradients update ---- #
            new_bwg = scan_fn_on2(
                hist_bwg,
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

            # ---- O(n) etrace gradients update ---- #
            new_xs, new_dfs = scan_fn_on(
                (hist_xs, hist_dfs),
                (
                    hid2weight_jac_single_or_multi_times[0],
                    hid2weight_jac_single_or_multi_times[1],
                    hid2hid_jac_single_or_multi_times,
                ),
            )[0]

        return new_xs, new_dfs, new_bwg

    def _solve_weight_gradients(
        self,
        running_index: int,
        etrace_h2w_at_t: Tuple,
        dl_to_hidden_groups: Sequence[jax.Array],
        weight_vals: Dict[Path, PyTree],
        dl_to_nonetws_at_t: Dict[Path, PyTree],
        dl_to_etws_at_t: Optional[Dict[Path, PyTree]],
    ):
        """
        Solve the weight gradients according to the eligibility trace data.

        Particularly, for each weight, we compute its gradients according to the batched weight gradients.
        """

        #
        # dl_to_hidden_groups:
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
        for relation in self.graph.hidden_param_op_relations:
            if _is_weight_need_full_grad(relation, self.mode):
                on2_weight_hidden_relations.append(relation)
            else:
                on_weight_hidden_relations.append(relation)

        # --- update the etrace weight gradients by the O(n) algorithm --- #

        _solve_IO_dim_weight_gradients(
            (xs, dfs),
            dG_weights,
            dl_to_hidden_groups,
            on_weight_hidden_relations,
            weight_vals,
            running_index,
            self.decay,
            self.mode,
        )

        # --- update the etrace weight gradients by the O(n^2) algorithm --- #

        _solve_param_dim_weight_gradients(
            wgrads,
            dG_weights,
            dl_to_hidden_groups,
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


ES_D_RTRL = IODimVjpAlgorithm
D_RTRL = ParamDimVjpAlgorithm
