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

from enum import Enum
from typing import Callable, Optional, Dict, Tuple, Sequence, Union

import brainstate
import brainunit as u
import jax
import numpy as np

from ._etrace_operators import (
    ETraceOp,
    ElemWiseOp,
)
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

X = brainstate.typing.ArrayLike
W = brainstate.typing.PyTree
Y = brainstate.typing.ArrayLike


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


class ETraceState(brainstate.HiddenState):
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

    value: brainstate.typing.ArrayLike

    def __init__(
        self,
        value: brainstate.typing.ArrayLike,
        name: Optional[str] = None
    ):
        self._check_value(value)
        super().__init__(value, name=name)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        Get the shape of the hidden state variable.

        This property returns the shape of the hidden state variable stored in the instance.
        It provides the dimensions of the array representing the hidden state.

        Returns:
            Tuple[int, ...]: A tuple representing the shape of the hidden state variable.
        """
        return self.value.shape

    @property
    def num_state(self) -> int:
        """
        Get the number of hidden states.

        This property returns the number of hidden states represented by the instance.
        For the `ETraceState` class, this is always 1, as it represents a single hidden state.

        Returns:
            int: The number of hidden states, which is 1 for this class.
        """
        return 1

    def _check_value(self, value: brainstate.typing.ArrayLike):
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
    value: brainstate.typing.ArrayLike
    name2index: Dict[str, int]

    def __init__(
        self,
        value: brainstate.typing.ArrayLike,
    ):
        value, name2index = self._check_value(value)
        self.name2index = name2index
        brainstate.ShortTermState.__init__(self, value)

    @property
    def varshape(self) -> Tuple[int, ...]:
        """
        Get the shape of each hidden state variable.

        This property returns the shape of the hidden state variables, excluding
        the last dimension which represents the number of hidden states.

        Returns:
            Tuple[int, ...]: A tuple representing the shape of each hidden state variable.
        """
        return self.value.shape[:-1]

    @property
    def num_state(self) -> int:
        """
        Get the number of hidden states.

        This property returns the number of hidden states represented by the last dimension
        of the value array.

        Returns:
            int: The number of hidden states.
        """
        return self.value.shape[-1]

    def _check_value(self, value) -> Tuple[brainstate.typing.ArrayLike, Dict[str, int]]:
        """
        Validates the input value for hidden states and returns a tuple containing
        the processed value and a dictionary mapping state names to indices.

        This function ensures that the input value is of a supported type and has
        the required dimensionality for hidden states. It also constructs a mapping
        from string representations of indices to their integer counterparts.

        Parameters
        ----------
        value (brainstate.typing.ArrayLike): The input value representing hidden states.
            It must be an instance of numpy.ndarray, jax.Array, or brainunit.Quantity
            with at least two dimensions.

        Returns
        -------
        Tuple[brainstate.typing.ArrayLike, Dict[str, int]]: A tuple containing:
            - The validated and possibly modified input value.
            - A dictionary mapping string representations of indices to integer indices.

        Raises
        ------
        TypeError: If the input value is not of a supported type.
        ValueError: If the input value does not have the required number of dimensions.
        """
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

    def get_value(self, item: int | str) -> brainstate.typing.ArrayLike:
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

    def set_value(self,
                  val: Dict[int | str, brainstate.typing.ArrayLike] | Sequence[brainstate.typing.ArrayLike]) -> None:
        """
        Set the value of the hidden state with the specified item.

        This method updates the hidden state values based on the provided dictionary or sequence.
        The values are set according to the indices or names specified in the input.

        Parameters
        ----------
        val (Dict[int | str, brainstate.typing.ArrayLike] | Sequence[brainstate.typing.ArrayLike]):
            A dictionary or sequence containing the new values for the hidden states.
            - If a dictionary, keys can be integers (indices) or strings (names) of the hidden states.
            - If a sequence, it is converted to a dictionary with indices as keys.

        Returns
        -------
        None: This method does not return any value. It updates the hidden state values in place.
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
    value: brainstate.typing.ArrayLike

    def __init__(
        self,
        value: Dict[str, brainstate.typing.ArrayLike] | Sequence[brainstate.typing.ArrayLike],
    ):
        value, name2unit, name2index = self._check_value(value)
        self.name2unit: Dict[str, u.Unit] = name2unit
        self.name2index: Dict[str, int] = name2index
        self.index2unit: Dict[int, u.Unit] = {i: v for i, v in enumerate(name2unit.values())}
        self.index2name: Dict[int, str] = {v: k for k, v in name2index.items()}
        brainstate.ShortTermState.__init__(self, value)

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
    ) -> Tuple[brainstate.typing.ArrayLike, Dict[str, u.Unit], Dict[str, int]]:
        """
        Validates and processes the input value to ensure it conforms to the expected format
        and structure for hidden states.

        This function checks if the input value is a dictionary or sequence of hidden states,
        verifies that all hidden states have the same shape, and extracts units and indices
        for each hidden state.

        Args:
            value (dict | Sequence): A dictionary or sequence representing hidden states.
                - If a sequence, it is converted to a dictionary with string indices as keys.
                - Each hidden state should be a numpy.ndarray, jax.Array, or brainunit.Quantity.

        Returns:
            Tuple[brainstate.typing.ArrayLike, Dict[str, u.Unit], Dict[str, int]]:
                - A stacked array of hidden state magnitudes.
                - A dictionary mapping hidden state names to their units.
                - A dictionary mapping hidden state names to their indices.

        Raises:
            TypeError: If any hidden state is not a numpy.ndarray, jax.Array, or brainunit.Quantity.
            ValueError: If hidden states do not have the same shape.
        """
        if isinstance(value, (tuple, list)):
            value = {str(i): v for i, v in enumerate(value)}
        assert isinstance(value, dict), (
            f'Currently, {self.__class__.__name__} only supports '
            f'dictionary/sequence of hidden states. But we got {type(value)}.'
        )
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

    def get_value(self, item: str | int) -> brainstate.typing.ArrayLike:
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

    def set_value(
        self,
        val: Dict[int | str, brainstate.typing.ArrayLike] | Sequence[brainstate.typing.ArrayLike]
    ) -> None:
        """
        Set the value of the hidden state with the specified item.

        This method updates the hidden state values based on the provided dictionary or sequence.
        The values are set according to the indices or names specified in the input.

        Parameters
        ----------
        val (Dict[int | str, brainstate.typing.ArrayLike] | Sequence[brainstate.typing.ArrayLike]):
            A dictionary or sequence containing the new values for the hidden states.
            - If a dictionary, keys can be integers (indices) or strings (names) of the hidden states.
            - If a sequence, it is converted to a dictionary with indices as keys.

        Returns
        -------
        None: This method does not return any value. It updates the hidden state values in place.
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
        if len(indices) == 0:
            raise ValueError(
                f'No hidden state is set. Please check the hidden state names or indices.'
            )
        if len(indices) == 1:
            indices = indices[0]
            values = values[0]
        else:
            indices = np.asarray(indices)
            values = u.math.stack(values, axis=-1)
        self.value = self.value.at[..., indices].set(values)


class ETraceParam(brainstate.ParamState):
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

    value: brainstate.typing.PyTree  # weight
    op: ETraceOp  # operator
    is_etrace: bool  # whether the operator is a true eligibility trace

    def __init__(
        self,
        weight: brainstate.typing.PyTree,
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

        This method applies the associated operator to the input data and the stored
        parameter value, performing the defined operation.

        Args:
            x (X): The input data on which the operator will be executed.

        Returns:
            Y: The result of applying the operator to the input data and the parameter value.
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
    value: brainstate.typing.PyTree  # weight
    op: ElemWiseOp  # operator

    def __init__(
        self,
        weight: brainstate.typing.PyTree,
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
        """
        Executes the associated operator on the stored weight.

        This method applies the operator to the weight of the element-wise parameter
        state, performing the defined element-wise operation.

        Returns:
            Y: The result of applying the operator to the weight.
        """
        return self.op(self.value)


class NonTempParam(brainstate.ParamState):
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
    value: brainstate.typing.PyTree  # weight

    def __init__(
        self,
        value: brainstate.typing.PyTree,
        op: Callable,
        name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(value, name=name)

        # operation
        if isinstance(op, ETraceOp):
            op = op.xw_to_y
        self.op = op

    def execute(self, x: jax.Array) -> jax.Array:
        """
        Executes the associated operator on the input data and the stored parameter value.

        Args:
            x (jax.Array): The input data on which the operator will be executed.

        Returns:
            jax.Array: The result of applying the operator to the input data and the parameter value.
        """
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
    value: brainstate.typing.PyTree  # weight

    def __init__(
        self,
        value: brainstate.typing.PyTree,
        op: Callable
    ):
        super().__init__()

        self.value = value
        if isinstance(op, ETraceOp):
            op = op.xw_to_y
        self.op = op

    def execute(self, x: brainstate.typing.ArrayLike) -> brainstate.typing.ArrayLike:
        """
        Executes the associated operator on the input data and the stored parameter value.

        Args:
            x (brainstate.typing.ArrayLike): The input data on which the operator will be executed.

        Returns:
            brainstate.typing.ArrayLike: The result of applying the operator to the input data and the parameter value.
        """
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
    value: brainstate.typing.PyTree  # weight

    def __init__(
        self,
        weight: brainstate.typing.PyTree,
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

    def execute(self) -> brainstate.typing.ArrayLike:
        """
        Executes the associated operator on the stored weight.

        This method applies the operator to the weight of the fake element-wise parameter
        state, simulating the behavior of an element-wise operation without computing gradients.

        Returns:
            brainstate.typing.ArrayLike: The result of applying the operator to the weight.
        """
        return self.op(None, self.value)
