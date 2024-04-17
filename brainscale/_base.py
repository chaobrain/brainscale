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
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Sequence, Optional, Tuple

import braincore as bc
from braincore.mixin import Mixin, DelayedInit, Delayed

__all__ = [
  # network related concepts.
  'ExplicitInOutSize',
  'ElementWiseBlock',
  'Sequential',
  'DnnLayer',
]


# -------------------------------------------------------------------------------------- #
# Network Related Concepts
# -------------------------------------------------------------------------------------- #


class ExplicitInOutSize(Mixin):
  """
  Mix-in class with the explicit input and output shape.

  Attributes
  ----------
  in_size: tuple[int]
    The input shape, without the batch size. This argument is important, since it is
    used to evaluate the shape of the output.
  out_size: tuple[int]
    The output shape, without the batch size.
  """
  __module__ = 'brainscale'

  _in_size: Optional[Tuple[int, ...]] = None
  _out_size: Optional[Tuple[int, ...]] = None

  @property
  def in_size(self) -> Tuple[int, ...]:
    if self._in_size is None:
      raise ValueError(f"The input shape is not set in this node: {self} ")
    return self._in_size

  @in_size.setter
  def in_size(self, in_size: Sequence[int]):
    self._in_size = tuple(in_size)

  @property
  def out_size(self) -> Tuple[int, ...]:
    if self._out_size is None:
      raise ValueError(f"The output shape is not set in this node: {self}")
    return self._out_size

  @out_size.setter
  def out_size(self, out_size: Sequence[int]):
    self._out_size = tuple(out_size)


class ElementWiseBlock(Mixin):
  """
  Mix-in class for element-wise modules.
  """
  __module__ = 'brainscale'


class Sequential(bc.Sequential, ExplicitInOutSize):
  """
  A sequential `input-output` module.

  Modules will be added to it in the order they are passed in the
  constructor. Alternatively, an ``dict`` of modules can be
  passed in. The ``update()`` method of ``Sequential`` accepts any
  input and forwards it to the first module it contains. It then
  "chains" outputs to inputs sequentially for each subsequent module,
  finally returning the output of the last module.

  The value a ``Sequential`` provides over manually calling a sequence
  of modules is that it allows treating the whole container as a
  single module, such that performing a transformation on the
  ``Sequential`` applies to each of the modules it stores (which are
  each a registered submodule of the ``Sequential``).

  What's the difference between a ``Sequential`` and a
  :py:class:`Container`? A ``Container`` is exactly what it
  sounds like--a container to store :py:class:`DynamicalSystem` s!
  On the other hand, the layers in a ``Sequential`` are connected
  in a cascading way.

  Examples
  --------

  >>> import braincore as bc
  >>> import brainscale as nn
  >>> import jax
  >>>
  >>> # composing ANN models
  >>> l = nn.Sequential(nn.Linear(100, 10),
  >>>                   jax.nn.relu,
  >>>                   nn.Linear(10, 2))
  >>> l(bc.random.random((256, 100)))
  >>>
  >>> # Using Sequential with Dict. This is functionally the
  >>> # same as the above code
  >>> l = nn.Sequential(l1=nn.Linear(100, 10),
  >>>                   l2=jax.nn.relu,
  >>>                   l3=nn.Linear(10, 2))
  >>> l(bc.random.random((256, 100)))

  Args:
    modules_as_tuple: The children modules.
    modules_as_dict: The children modules.
    name: The object name.
    mode: The object computing context/mode. Default is ``None``.
  """
  __module__ = 'brainscale'

  def __init__(self, first: ExplicitInOutSize, *modules_as_tuple, **modules_as_dict):
    super().__init__()

    assert isinstance(first, ExplicitInOutSize)
    in_size = first.out_size

    tuple_modules = []
    for module in modules_as_tuple:
      module, in_size = self._format_module(module, in_size)
      tuple_modules.append(module)

    dict_modules = dict()
    for key, module in modules_as_dict.items():
      module, in_size = self._format_module(module, in_size)
      dict_modules[key] = module

    super().__init__(first, *tuple_modules, **dict_modules)

    # the input and output shape
    self.in_size = tuple(first.in_size)
    self.out_size = tuple(in_size)

  def _format_module(self, module, in_size):
    if isinstance(module, DelayedInit):
      module = module(in_size=in_size)
      assert isinstance(module, ExplicitInOutSize)
      out_size = module.out_size
    elif isinstance(module, ElementWiseBlock):
      out_size = in_size
    elif isinstance(module, ExplicitInOutSize):
      out_size = module.out_size
    else:
      raise TypeError(f"Unsupported type {type(module)}. ")
    return module, out_size


class DnnLayer(bc.Module, ExplicitInOutSize, Delayed):
  """
  A DNN layer.
  """
  __module__ = 'brainscale'

  def __repr__(self):
    return f"{self.__class__.__name__}(in_size={self.in_size}, out_size={self.out_size})"
