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

from brainscale._etrace_algorithms import (
    ETraceAlgorithm,
    EligibilityTraceData,
)
from brainscale._etrace_compiler_graph import (
    CompiledGraph,
)
from brainscale._etrace_compiler_hid_param_op import (
    HiddenParamOpRelation,
)
from brainscale._etrace_compiler_hidden_group import (
    HiddenGroup,
)
from brainscale._etrace_compiler_hidden_pertubation import (
    HiddenPerturbation,
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
from brainscale._etrace_graph import (
    ETraceGraphExecutor,
)
from brainscale._etrace_input_data import (
    SingleStepData,
    MultiStepData,
)
from brainscale._etrace_operators import (
    ETraceOp,
    MatMulOp,
    ElemWiseOp,
)
from brainscale._etrace_vjp_algorithms import (
    ETraceVjpAlgorithm,
    IODimVjpAlgorithm,
    ParamDimVjpAlgorithm,
    HybridDimVjpAlgorithm,
)
from brainscale._etrace_vjp_compiler import (
    CompiledVjpGraph
)
from brainscale._grad_exponential import (
    GradExpon,
)
from brainscale._misc import (
    CompilationError,
    NotSupportedError,
)
from . import nn
# Added 2024-10-27
from ._misc import deprecation_getattr

_deprecations = {k: (f"'brainscale.{k}' has been moved into brainscale.nn.{k}", getattr(nn, k)) for k in nn.__all__}
__getattr__ = deprecation_getattr(__name__, _deprecations)
del deprecation_getattr
