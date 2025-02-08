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
# Date: 2024-04-03
# Copyright: 2024, Chaoming Wang
#
# Refinement History:
# [2025-02-06]
#   - Add the `ETraceTreeState` and `ETraceGroupState` for the multiple hidden states.
#   - Add the `ElemWiseParam` for the element-wise eligibility trace parameters.
#   - Remove previous `ETraceParam` and `ETraceParamOp`
#   - Unify the `ETraceParam` and `ETraceParamOp` into the `ETraceParam`
#   - Add the `FakeETraceParam` and `FakeElemWiseParam` for the fake parameter states.
#
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Dict, Tuple, Sequence

import brainstate as bst
import brainunit as u
import jax
import numpy as np

from ._etrace_operators import ETraceOp, ElemWiseOp
from ._misc import BaseEnum

__all__ = [
    # eligibility trace states
    'ETraceState',  # single hidden state for the etrace-based learning
    'ETraceGroupState',  # multiple hidden state for the etrace-based learning
    'ETraceTreeState',  # dictionary of hidden states for the etrace-based learning

    # eligibility trace parameters and operations
    'ETraceParam',  # the parameter and operator for the etrace-based learning, combining ETraceParam and ETraceOp
    'NonTempParam',  # the parameter state with an associated operator without temporal dependent gradient learning

    # element-wise eligibility trace parameters
    'ElemWiseParam',  # the element-wise weight parameter for the etrace-based learning

    # fake parameter state
    'FakeETraceParam',  # the fake parameter state with an associated operator
    'FakeElemWiseParam',  # the fake element-wise parameter state with an associated operator
]

X = bst.typing.ArrayLike
W = bst.typing.PyTree
Y = bst.typing.ArrayLike


class ETraceGrad(BaseEnum):
    """
    The Gradient Type for the Eligibility Trace.

    This defines how the weight gradient is computed in the eligibility trace-based learning.

    - `full`: The full eligibility trace gradient is computed.
    - `approx`: The approximated eligibility trace gradient is computed.
    - `adaptive`: The adaptive eligibility trace gradient is computed.

    """
    full = 'full'
    approx = 'approx'
    adaptive = 'adaptive'


class ETraceState(bst.ShortTermState):
    """
    The Hidden State for Eligibility Trace-based Learning.

    .. note::

        Currently, the hidden state only supports `jax.Array` or `brainunit.Quantity`.
        This means that each instance of :py:class:`ETraceState` should define
        single hidden variable.

        If you want to define multiple hidden variables within a single instance of
        :py:class:`ETraceState`, you can try :py:class:`ETraceGroupState` or
        :py:class:`ETraceTreeState` instead.

    Args:
        value: The value of the hidden state.
               Currently only support a `jax.Array` or `brainunit.Quantity`.
        name: The name of the hidden state.
    """
    __module__ = 'brainscale'

    value: bst.typing.ArrayLike

    def __init__(
        self,
        value: bst.typing.PyTree,
        name: Optional[str] = None
    ):
        self._check_value(value)
        super().__init__(value, name=name)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        The variable shape of the hidden state.
        """
        return self.value.shape

    @property
    def num_state(self) -> int:
        return 1

    def _check_value(self, value: bst.typing.ArrayLike):
        if not isinstance(value, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(
                f'Currently, {ETraceState.__name__} only supports '
                f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                f'But we got {type(value)}.'
            )


class ETraceGroupState(ETraceState):
    """
    A group of multiple hidden states for eligibility trace-based learning.

    This class is used to define multiple hidden states within a single instance
    of :py:class:`ETraceState`. Normally, you should define multiple instances
    of :py:class:`ETraceState` to represent multiple hidden states. But
    :py:class:`ETraceGroupState` let your define multiple hidden states within
    a single instance.

    The following is the way to initialize the hidden states.

    .. code-block:: python

        import brainunit as u
        value = np.random.randn(10, 10, 5) * u.mV
        state = ETraceGroupState(value)

    Then, you can retrieve the hidden state value with the following method.

    .. code-block:: python

        state.get_value(0)  # get the first hidden state
        # or
        state.get_value('0')  # get the hidden state with the name '0'

    You can write the hidden state value with the following method.

    .. code-block:: python

        state.set_value({0: np.random.randn(10, 10) * u.mV})  # set the first hidden state
        # or
        state.set_value({'0': np.random.randn(10, 10) * u.mV})  # set the hidden state with the name '0'
        # or
        state.value = np.random.randn(10, 10, 5) * u.mV  # set all hidden state value

    Args:
        value: The values of the hidden states. It can be a sequence of hidden states,
            or a single hidden state with the last dimension as the number of hidden states,
            or a dictionary of hidden states.
    """

    __module__ = 'brainscale'
    value: bst.typing.ArrayLike
    name2index: Dict[str, int]

    def __init__(
        self,
        value: bst.typing.PyTree,
    ):
        value, name2index = self._check_value(value)
        self.name2index = name2index
        bst.ShortTermState.__init__(self, value)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        The shape of each hidden state variable.
        """
        return self.value.shape[:-1]

    @property
    def num_state(self) -> int:
        return self.value.shape[-1]

    def _check_value(self, value) -> Tuple[bst.typing.ArrayLike, Dict[str, int]]:
        if not isinstance(value, (np.ndarray, jax.Array, u.Quantity)):
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                f'But we got {type(value)}.'
            )
        if value.ndim < 2:
            raise ValueError(
                f'Currently, {self.__class__.__name__} only supports '
                f'hidden states with more than 2 dimensions, where the last '
                f'dimension is the number of state size and the other dimensions '
                f'are the hidden shape. '
                f'But we got {value.ndim} dimensions.'
            )
        name2index = {str(i): i for i in range(value.shape[-1])}
        return value, name2index

    def get_value(self, item: int | str) -> bst.typing.ArrayLike:
        """
        Get the value of the hidden state with the item.

        Args:
            item: int or str. The index of the hidden state.
                - If int, the index of the hidden state.
                - If str, the name of the hidden state.
        Returns:
            The value of the hidden state.
        """
        if isinstance(item, int):
            assert item < self.value.shape[-1], (f'Index {item} out of range. '
                                                 f'The maximum index is {self.value.shape[-1] - 1}.')
            return self.value[..., item]
        elif isinstance(item, str):
            assert item in self.name2index, (f'Hidden state name {item} not found. '
                                             f'Please check the hidden state names.')
            index = self.name2index[item]
            return self.value[..., index]
        else:
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'int or str for getting the hidden state. '
                f'But we got {type(item)}.'
            )

    def set_value(self, val: Dict[int | str, bst.typing.ArrayLike] | Sequence[bst.typing.ArrayLike]) -> None:
        """
        Set the value of the hidden state with the item.
        """
        if isinstance(val, (tuple, list)):
            val = {i: v for i, v in enumerate(val)}
        assert isinstance(val, dict), (f'Currently, {self.__class__.__name__}.set_value() only supports '
                                       f'dictionary of hidden states. But we got {type(val)}.')
        indices = []
        values = []
        for k, v in val.items():
            if isinstance(k, str):
                k = self.name2index[k]
            assert isinstance(k, int), (f'Key {k} should be int or str. '
                                        f'But we got {type(k)}.')
            assert v.shape == self.varshape, (f'The shape of the hidden state should be {self.varshape}. '
                                              f'But we got {v.shape}.')
            indices.append(k)
            values.append(v)
        values = u.math.stack(values, axis=-1)
        self.value = self.value.at[..., indices].set(values)


class ETraceTreeState(ETraceGroupState):
    """
    A pytree of multiple hidden states for eligibility trace-based learning.

    .. note::

        The value in this state class behaves likes a dictionary/sequence of hidden states.
        However, the state is actually stored as a single dimensionless array.

    There are two ways to define the hidden states.

    1. The first is to define a sequence of hidden states.

    .. code-block:: python

        import brainunit as u
        value = [np.random.randn(10, 10) * u.mV,
                 np.random.randn(10, 10) * u.mA,
                 np.random.randn(10, 10) * u.mS]
        state = ETraceTreeState(value)

    Then, you can retrieve the hidden state value with the following method.

    .. code-block:: python

        state.get_value(0)  # get the first hidden state
        # or
        state.get_value('0')  # get the hidden state with the name '0'

    You can write the hidden state value with the following method.

    .. code-block:: python

        state.set_value({0: np.random.randn(10, 10) * u.mV})  # set the first hidden state
        # or
        state.set_value({'1': np.random.randn(10, 10) * u.mA})  # set the hidden state with the name '1'
        # or
        state.set_value([np.random.randn(10, 10) * u.mV,
                         np.random.randn(10, 10) * u.mA,
                         np.random.randn(10, 10) * u.mS])  # set all hidden state value
        # or
        state.set_value({
            0: np.random.randn(10, 10) * u.mV,
            1: np.random.randn(10, 10) * u.mA,
            2: np.random.randn(10, 10) * u.mS
        })  # set all hidden state value

    2. The second is to define a dictionary of hidden states.

    .. code-block:: python

        import brainunit as u
        value = {'v': np.random.randn(10, 10) * u.mV,
                 'i': np.random.randn(10, 10) * u.mA,
                 'g': np.random.randn(10, 10) * u.mS}
        state = ETraceTreeState(value)

    Then, you can retrieve the hidden state value with the following method.

    .. code-block:: python

        state.get_value('v')  # get the hidden state with the name 'v'
        # or
        state.get_value('i')  # get the hidden state with the name 'i'

    You can write the hidden state value with the following method.

    .. code-block:: python

        state.set_value({'v': np.random.randn(10, 10) * u.mV})  # set the hidden state with the name 'v'
        # or
        state.set_value({'i': np.random.randn(10, 10) * u.mA})  # set the hidden state with the name 'i'
        # or
        state.set_value([np.random.randn(10, 10) * u.mV,
                         np.random.randn(10, 10) * u.mA,
                         np.random.randn(10, 10) * u.mS])  # set all hidden state value
        # or
        state.set_value({
            'v': np.random.randn(10, 10) * u.mV,
            'g': np.random.randn(10, 10) * u.mA,
            'i': np.random.randn(10, 10) * u.mS
        })  # set all hidden state value

    .. note::

        Avoid using ``ETraceTreeState.value`` to get the state value, or
        ``ETraceTreeState.value =`` to assign the state value.

        Instead, use ``ETraceTreeState.get_value()`` and ``ETraceTreeState.set_value()``.
        This is because ``.value`` loss hidden state units and other information,
        and it is only dimensionless data.

        This design aims to ensure that any etrace hidden state has only one array.


    Args:
        value: The values of the hidden states.
    """

    __module__ = 'brainscale'
    value: bst.typing.ArrayLike

    def __init__(
        self,
        value: Dict[str, bst.typing.ArrayLike] | Sequence[bst.typing.ArrayLike],
    ):
        value, name2unit, name2index = self._check_value(value)
        self.name2unit: Dict[str, u.Unit] = name2unit
        self.name2index: Dict[str, int] = name2index
        self.index2unit: Dict[int, u.Unit] = {i: v for i, v in enumerate(name2unit.values())}
        self.index2name: Dict[int, str] = {v: k for k, v in name2index.items()}
        bst.ShortTermState.__init__(self, value)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        The shape of each hidden state variable.
        """
        return self.value.shape[:-1]

    @property
    def num_state(self) -> int:
        """
        The number of hidden states.
        """
        assert self.value.shape[-1] == len(self.name2index), (
            f'The number of hidden states '
            f'is not equal to the number of hidden state names.'
        )
        return self.value.shape[-1]

    def _check_value(
        self,
        value: dict | Sequence
    ) -> Tuple[bst.typing.ArrayLike, Dict[str, u.Unit], Dict[str, int]]:
        if isinstance(value, (tuple, list)):
            value = {str(i): v for i, v in enumerate(value)}
        assert isinstance(value, dict), (f'Currently, {self.__class__.__name__} only supports '
                                         f'dictionary/sequence of hidden states. But we got {type(value)}.')
        shapes = []
        for k, v in value.items():
            if not isinstance(v, (np.ndarray, jax.Array, u.Quantity)):
                raise TypeError(
                    f'Currently, {self.__class__.__name__} only supports '
                    f'numpy.ndarray, jax.Array or brainunit.Quantity. '
                    f'But we got {type(v)} for key {k}.'
                )
            shapes.append(v.shape)
        if len(set(shapes)) > 1:
            info = {k: v.shape for k, v in value.items()}
            raise ValueError(
                f'Currently, {self.__class__.__name__} only supports '
                f'hidden states with the same shape. '
                f'But we got {info}.'
            )
        name2unit = {k: u.get_unit(v) for k, v in value.items()}
        name2index = {k: i for i, k in enumerate(value.keys())}
        value = u.math.stack([u.get_magnitude(v) for v in value.values()], axis=-1)
        return value, name2unit, name2index

    def get_value(self, item: str | int) -> bst.typing.ArrayLike:
        """
        Get the value of the hidden state with the key.

        Args:
            item: The key of the hidden state.
                - If int, the index of the hidden state.
                - If str, the name of the hidden state.
        """
        if isinstance(item, int):
            assert item < self.value.shape[-1], (f'Index {item} out of range. '
                                                 f'The maximum index is {self.value.shape[-1] - 1}.')
            val = self.value[..., item]
        elif isinstance(item, str):
            assert item in self.name2index, (f'Hidden state name {item} not found. '
                                             f'Please check the hidden state names.')
            item = self.name2index[item]
            val = self.value[..., item]
        else:
            raise TypeError(
                f'Currently, {self.__class__.__name__} only supports '
                f'int or str for getting the hidden state. '
                f'But we got {type(item)}.'
            )
        if self.index2unit[item].dim.is_dimensionless:
            return val
        else:
            return val * self.index2unit[item]

    def set_value(self, val: Dict[int | str, bst.typing.ArrayLike] | Sequence[bst.typing.ArrayLike]) -> None:
        """
        Set the value of the hidden state with the item.
        """
        if isinstance(val, (tuple, list)):
            val = {i: v for i, v in enumerate(val)}
        assert isinstance(val, dict), (f'Currently, {self.__class__.__name__}.set_value() only supports '
                                       f'dictionary of hidden states. But we got {type(val)}.')
        indices = []
        values = []
        for index, v in val.items():
            if isinstance(index, str):
                index = self.name2index[index]
            assert isinstance(index, int), (f'Key {index} should be int or str. '
                                            f'But we got {type(index)}.')
            assert v.shape == self.varshape, (f'The shape of the hidden state should be {self.varshape}. '
                                              f'But we got {v.shape}.')
            indices.append(index)
            values.append(u.Quantity(v).to(self.index2unit[index]).mantissa)
        values = u.math.stack(values, axis=-1)
        self.value = self.value.at[..., indices].set(values)


class ETraceParam(bst.ParamState):
    """
    The Eligibility Trace Weight and its Associated Operator.

    .. note::

        Although one weight is defined as :py:class:`ETraceParam`,
        whether eligibility traces are used for training with temporal
        dependencies depends on the final compilation result of the
        compiler regarding this parameter. If no hidden states are
        found to associate this parameter, the training based on
        eligibility traces will not be performed.
        Then, this parameter will perform the same as :py:class:`NonTempParam`.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ETraceOp`.
        grad: The gradient type for the ETrace. Default is `adaptive`.
        name: The name of the weight-operator.
    """
    __module__ = 'brainscale'

    value: bst.typing.PyTree  # weight
    op: ETraceOp  # operator
    is_etrace: bool  # whether the operator is a true eligibility trace

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: ETraceOp,
        grad: Optional[str | Enum] = None,
        name: Optional[str] = None
    ):
        # weight value
        super().__init__(weight, name=name)

        # gradient
        if grad is None:
            grad = 'adaptive'
        self.gradient = ETraceGrad.get(grad)

        # operation
        assert isinstance(op, ETraceOp), (
            f'op should be ETraceOp. '
            f'But we got {type(op)}.'
        )
        self.op = op

        # check if the operator is not an eligibility trace
        self.is_etrace = True

    def execute(self, x: X) -> Y:
        """
        Execute the operator with the input.
        """
        return self.op(x, self.value)


class ElemWiseParam(ETraceParam):
    r"""
    The Element-wise Eligibility Trace Weight and its Associated Operator.

    .. note::

        The ``element-wise`` is called with the correspondence to the hidden state.
        That means the operator performs element-wise operations with the hidden state.

    It supports all element-wise operations for the eligibility trace-based learning.
    For example, if the parameter weight has the shape with the same as the hidden state,

        $$
        I = \theta_1 * h_1
        $$

    where $\theta_1 \in \mathbb{R}^H$ is the weight and $h_1 \in \mathbb{R}^H$ is the
    hidden state. The element-wise operation is defined as:

    .. code-block:: python

       op = ElemWiseParam(weight, op=lambda w: w)

    If the parameter weight is a scalar,

        $$
        I = \theta * h
        $$

    where $\theta \in \mathbb{R}$ is the weight and $h \in \mathbb{R}^H$ is the hidden state.
    Then the element-wise operation can be defined as:

    .. code-block:: python

         h = 100  # hidden size
         op = ElemWiseParam(weight, op=lambda w: w * jax.numpy.ones(h))

    Other element-wise operations can be defined in the same way.

    Moreover, :py:class:`ElemWiseParam` support a pytree of element-wise parameters. For example,
    if the mathematical operation is defined as:

        $$
        I = \theta_1 * h_1 + \theta_2 * h_2
        $$

    where $\theta_1 \in \mathbb{R}^H$ and $\theta_2 \in \mathbb{R}^H$ are the weights and
    $h_1 \in \mathbb{R}^H$ and $h_2 \in \mathbb{R}^H$ are the hidden states. The element-wise
    operation can be defined as:

    .. code-block:: python

        op = ElemWiseParam(
            weight={'w1': weight1, 'w2': weight2},
            op=lambda w: w['w1'] * h1 + w['w2'] * h2
        )

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ElemWiseOp`.
        name: The name of the weight-operator.
    """
    __module__ = 'brainscale'
    value: bst.typing.PyTree  # weight
    op: ElemWiseOp  # operator

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: ElemWiseOp | Callable[[W], Y] = (lambda w: w),
        name: Optional[str] = None,
    ):
        if not isinstance(op, ElemWiseOp):
            op = ElemWiseOp(op)
        assert isinstance(op, ElemWiseOp), (
            f'op should be ElemWiseOp. '
            f'But we got {type(op)}.'
        )
        super().__init__(
            weight,
            op=op,
            grad=ETraceGrad.full,
            name=name
        )

    def execute(self) -> Y:
        return self.op(self.value)


class NonTempParam(bst.ParamState):
    r"""
    The Parameter State with an Associated Operator with no temporal dependent gradient learning.

    This class behaves the same as :py:class:`ETraceParam`, but will not build the
    eligibility trace graph when using online learning. Therefore, in a sequence
    learning task, the weight gradient can only be computed with the spatial gradients.
    That is,

        $$
        \nabla \theta = \sum_t \partial L^t / \partial \theta^t
        $$

    Instead, the gradient of the weight $\theta$ which is labeled as :py:class:`ETraceParam` is
    computed as:

        $$
        \nabla \theta = \sum_t \partial L^t / \partial \theta = \sum_t \sum_i^t \partial L^t / \partial \theta^i
        $$

    Args:
      value: The value of the parameter.
      op: The operator for the parameter. See `ETraceOp`.
    """
    __module__ = 'brainscale'
    op: Callable[[X, W], Y]  # operator
    value: bst.typing.PyTree  # weight

    def __init__(
        self,
        value: bst.typing.PyTree,
        op: Callable,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(value, name=name)

        # operation
        if isinstance(op, ETraceOp):
            op = op.xy_to_w
        self.op = op

    def execute(self, x: jax.Array) -> jax.Array:
        return self.op(x, self.value)


class FakeETraceParam(object):
    """
    The Parameter State with an Associated Operator that does not require to compute gradients.

    This class corresponds to the :py:class:`NonTempParam` and :py:class:`ETraceParam` but does
    not require to compute gradients. It has the same usage interface with :py:class:`NonTempParam`
    and :py:class:`ETraceParam`.

    Args:
      value: The value of the parameter.
      op: The operator for the parameter.
    """
    __module__ = 'brainscale'
    op: Callable[[X, W], Y]  # operator
    value: bst.typing.PyTree  # weight

    def __init__(
        self,
        value: bst.typing.PyTree,
        op: Callable
    ):
        super().__init__()

        self.value = value
        if isinstance(op, ETraceOp):
            op = op.xy_to_w
        self.op = op

    def execute(self, x: bst.typing.ArrayLike) -> bst.typing.ArrayLike:
        return self.op(x, self.value)


class FakeElemWiseParam(object):
    """
    The fake element-wise parameter state with an associated operator.

    This class corresponds to the :py:class:`ElemWiseParam` but does not require to compute gradients.
    It has the same usage interface with :py:class:`ElemWiseParam`. For usage, please see
    :py:class:`ElemWiseParam`.

    Args:
        weight: The weight of the ETrace.
        op: The operator for the ETrace. See :py:class:`ElemWiseOp`.
        name: The name of the weight-operator.
    """
    __module__ = 'brainscale'
    op: Callable[[W], Y]  # operator
    value: bst.typing.PyTree  # weight

    def __init__(
        self,
        weight: bst.typing.PyTree,
        op: ElemWiseOp | Callable[[W], Y] = (lambda w: w),
        name: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(op, ETraceOp):
            assert isinstance(op, ElemWiseOp), (
                f'op should be ElemWiseOp. '
                f'But we got {type(op)}.'
            )
            op = op.xw_to_y
        self.op = op
        self.value = weight
        self.name = name

    def execute(self) -> bst.typing.ArrayLike:
        return self.op(self.value)
