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
# ==============================================================================

# -*- coding: utf-8 -*-


__version__ = "0.0.11"
__version_info__ = (0, 0, 11)


from brainscale._etrace_algorithms import (
    ETraceAlgorithm,
    EligibilityTrace,
)
from brainscale._etrace_compiler_graph import (
    ETraceGraph,
    compile_etrace_graph,
)
from brainscale._etrace_compiler_hid_param_op import (
    HiddenParamOpRelation,
    find_hidden_param_op_relations_from_minfo,
    find_hidden_param_op_relations_from_module,
)
from brainscale._etrace_compiler_hidden_group import (
    HiddenGroup,
    find_hidden_groups_from_minfo,
    find_hidden_groups_from_module,
)
from brainscale._etrace_compiler_hidden_pertubation import (
    HiddenPerturbation,
    add_hidden_perturbation_from_minfo,
    add_hidden_perturbation_in_module,
)
from brainscale._etrace_compiler_module_info import (
    ModuleInfo,
    extract_module_info,
)
from brainscale._etrace_concepts import (
    # state
    ETraceState,
    ETraceGroupState,
    ETraceTreeState,
    # parameter
    ETraceParam,
    ElemWiseParam,
    NonTempParam,
    # fake parameter
    FakeETraceParam,
    FakeElemWiseParam,
)
from brainscale._etrace_graph_executor import (
    ETraceGraphExecutor,
)
from brainscale._etrace_input_data import (
    SingleStepData,
    MultiStepData,
)
from brainscale._etrace_operators import (
    ETraceOp,
    ElemWiseOp,
    MatMulOp,
    LoraOp,
    ConvOp,
    SpMatMulOp,
    stop_param_gradients,
)
from brainscale._etrace_vjp_algorithms import (
    ETraceVjpAlgorithm,
    IODimVjpAlgorithm, ES_D_RTRL,
    ParamDimVjpAlgorithm, D_RTRL,
    HybridDimVjpAlgorithm,
)
from brainscale._etrace_vjp_graph_executor import (
    ETraceVjpGraphExecutor,
)
from brainscale._grad_exponential import (
    GradExpon,
)
from brainscale._misc import (
    CompilationError,
    NotSupportedError,
)
from . import nn
