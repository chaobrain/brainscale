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


from __future__ import annotations

from enum import Enum
from typing import Sequence

import brainstate as bst

git_issue_addr = 'https://github.com/brainpy/brainscale/issues'


def state_traceback(states: Sequence[bst.State]):
  """
  Traceback the states of the brain model.

  Parameters
  ----------
  states : Sequence[bst.State]
    The states of the brain model.

  Returns
  -------
  str
    The traceback information of the states.
  """
  state_info = []
  for i, state in enumerate(states):
    state_info.append(f'State {i}: {state}\n'
                      f'defined at \n'
                      f'{state.source_info.traceback}\n')
  return '\n'.join(state_info)


def set_module_as(module: str = 'brainscale'):
  def wrapper(fun: callable):
    fun.__module__ = module
    return fun

  return wrapper


class BaseEnum(Enum):
  @classmethod
  def get_by_name(cls, name: str):
    for item in cls:
      if item.name == name:
        return item
    raise ValueError(f'Cannot find the {cls.__name__} type {name}.')

  @classmethod
  def get(cls, type_: str | Enum):
    if isinstance(type_, cls):
      return type_
    elif isinstance(type_, str):
      return cls.get_by_name(type_)
    else:
      raise ValueError(f'Cannot find the {cls.__name__} type {type_}.')
