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
# ==============================================================================

# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Any, Optional

import brainstate as bst

from ._etrace_compiler_graph import CompiledGraph
from ._etrace_concepts import ETraceState
from ._etrace_graph import ETraceGraphExecutor
from ._typing import Path

__all__ = [
    'ETraceAlgorithm',
    'EligibilityTraceData',
]


class EligibilityTraceData(bst.ShortTermState):
    """
    The temporary data for the eligibility trace algorithm computation.
    """
    pass


class ETraceAlgorithm(bst.nn.Module):
    r"""
    The base class for the eligibility trace algorithm.

    Parameters:
    -----------
    model: brainstate.nn.Module
        The model function, which receives the input arguments and returns the model output.
    name: str, optional
        The name of the etrace algorithm.

    Attributes:
    -----------
    graph: ETraceGraphExecutor
        The etrace graph.
    param_states: Dict[Hashable, bst.ParamState]
        The weight states.
    hidden_states: Dict[Hashable, ETraceState]
        The hidden states.
    other_states: Dict[Hashable, bst.State]
        The other states.
    is_compiled: bool
        Whether the etrace algorithm has been compiled.
    running_index: bst.ParamState[int]
        The running index.
    """
    __module__ = 'brainscale'

    def __init__(
        self,
        model: bst.nn.Module,
        graph: ETraceGraphExecutor,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # the model
        if not isinstance(model, bst.nn.Module):
            raise ValueError(
                f'The model should be a brainstate.nn.Module, this can help us to '
                f'better obtain the program structure. But we got {type(model)}.'
            )

        # the graph
        if not isinstance(graph, ETraceGraphExecutor):
            raise ValueError(
                f'The graph should be a ETraceGraphExecutor, this can help us to '
                f'better obtain the program structure. But we got {type(graph)}.'
            )
        self.graph = graph

        # The flag to indicate whether the etrace algorithm has been compiled
        self.is_compiled = False

        # the running index
        self.running_index = bst.LongTermState(0)

        # other states
        self._param_states = None
        self._hidden_states = None
        self._other_states = None

    @property
    def compiled(self) -> CompiledGraph:
        return self.graph.compiled

    @property
    def param_states(self) -> bst.util.FlattedDict[Path, bst.ParamState]:
        """
        Get the parameter weight states.
        """
        if self._param_states is None:
            self._split_state()
        return self._param_states

    @property
    def hidden_states(self) -> bst.util.FlattedDict[Path, ETraceState]:
        """
        Get the hidden states.
        """
        if self._hidden_states is None:
            self._split_state()
        return self._hidden_states

    @property
    def other_states(self) -> bst.util.FlattedDict[Path, bst.State]:
        """
        Get the other states.
        """
        if self._other_states is None:
            self._split_state()
        return self._other_states

    def _split_state(self):
        # --- the state separation --- #
        #
        # [NOTE]
        #
        # The `ETraceGraphExecutor` and the following states suggests that
        # `ETraceAlgorithm` depends on the states we created in the
        # `ETraceGraphExecutor`, including:
        #
        #   - the weight states, which is invariant during the training process
        #   - the hidden states, the recurrent states, which may be changed between different training epochs
        #   - the other states, which may be changed between different training epochs
        (
            self._param_states,
            self._hidden_states,
            self._other_states
        ) = self.compiled.model_retrieved_states.split(bst.ParamState, ETraceState, ...)

    def compile_graph(self, *args) -> None:
        r"""
        Compile the eligibility trace graph of the relationship between etrace weights, states and operators.

        The compilation process includes:
        - building the etrace graph
        - separating the states
        - initializing the etrace states

        Args:
            *args: the input arguments.
        """

        if not self.is_compiled:
            # --- the model etrace graph -- #
            self.graph.compile_graph(*args)

            # --- the initialization of the states --- #
            self.init_etrace_state(*args)

            # mark the graph is compiled
            self.is_compiled = True

    @property
    def path_to_states(self) -> bst.util.FlattedDict[Path, bst.State]:
        """
        Get the path to the states.
        """
        return self.graph.path_to_states

    @property
    def state_id_to_path(self) -> Dict[int, Path]:
        """
        Get the state ID to the path.
        """
        return self.graph.state_id_to_path

    def show_graph(self) -> None:
        """
        Show the etrace graph.
        """
        return self.graph.show_graph()

    def __call__(self, *args) -> Any:
        """
        Update the model and the eligibility trace states.
        """
        return self.update(*args)

    def update(self, *args) -> Any:
        """
        Update the model and the eligibility trace states.

        Args:
            *args: the input arguments.
        """
        raise NotImplementedError

    def init_etrace_state(self, *args, **kwargs) -> None:
        """
        Initialize the eligibility trace states of the etrace algorithm.

        This method is needed after compiling the etrace graph. See `.compile_graph()` for the details.
        """
        raise NotImplementedError

    def get_etrace_of(self, weight: bst.ParamState | Path) -> Any:
        """
        Get the eligibility trace of the given weight.

        Parameters
        ----------
        weight: brainstate.ParamState, int
          The parameter weight.

        Returns
        -------
        out: Any
          The eligibility trace.
        """
        raise NotImplementedError
