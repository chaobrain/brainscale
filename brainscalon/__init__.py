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

__version__ = "0.0.4"

from brainscalon._etrace_algorithms import (
    ETraceAlgorithm,
    EligibilityTrace,
)
from brainscalon._etrace_compiler_graph import (
    ETraceGraph,
    compile_etrace_graph,
)
from brainscalon._etrace_compiler_hid_param_op import (
    HiddenParamOpRelation,
    find_hidden_param_op_relations_from_module,
)
from brainscalon._etrace_compiler_hidden_group import (
    HiddenGroup,
    find_hidden_groups_from_module,
)
from brainscalon._etrace_compiler_hidden_pertubation import (
    HiddenPerturbation,
    add_hidden_perturbation_in_module,
)
from brainscalon._etrace_compiler_module_info import (
    ModuleInfo,
    extract_module_info,
)
from brainscalon._etrace_concepts import (
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
from brainscalon._etrace_graph_executor import (
    ETraceGraphExecutor,
)
from brainscalon._etrace_input_data import (
    SingleStepData,
    MultiStepData,
)
from brainscalon._etrace_operators import (
    ETraceOp,
    ElemWiseOp,
    MatMulOp,
    LoraOp,
    ConvOp,
    SpMVOp,
    stop_param_gradients,
)
from brainscalon._etrace_vjp_algorithms import (
    ETraceVjpAlgorithm,
    IODimVjpAlgorithm,
    ParamDimVjpAlgorithm,
    HybridDimVjpAlgorithm,
)
from brainscalon._etrace_vjp_graph_executor import (
    ETraceVjpGraphExecutor,
)
from brainscalon._grad_exponential import (
    GradExpon,
)
from brainscalon._misc import (
    CompilationError,
    NotSupportedError,
)
from . import nn

# # Added 2024-10-27
# from ._misc import deprecation_getattr
#
# _deprecations = {k: (f"'brainscalon.{k}' has been moved into brainscalon.nn.{k}", getattr(nn, k)) for k in nn.__all__}
# __getattr__ = deprecation_getattr(__name__, _deprecations)
# del deprecation_getattr
