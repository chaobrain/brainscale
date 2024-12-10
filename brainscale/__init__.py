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

__version__ = "0.0.3"

from brainscale._etrace_compiler import *
from brainscale._etrace_compiler import __all__ as etrace_compiler
from brainscale._etrace_algorithms import *
from brainscale._etrace_algorithms import __all__ as etrace_algorithms
from brainscale._etrace_graph import *
from brainscale._etrace_graph import __all__ as etrace_compiler_all
from brainscale._etrace_concepts import *
from brainscale._etrace_concepts import __all__ as concepts_all
from brainscale._etrace_operators import *
from brainscale._etrace_operators import __all__ as operators_all
from brainscale._misc import *
from brainscale._misc import __all__ as misc_all
from . import nn

__all__ = (
    ['nn']
    + concepts_all
    + etrace_algorithms
    + etrace_compiler_all
    + etrace_compiler
    + operators_all
    + misc_all
)

del (
    concepts_all,
    etrace_algorithms,
    etrace_compiler_all,
    etrace_compiler,
    operators_all,
    misc_all,
)

# Added 2024-10-27
from ._misc import deprecation_getattr

_deprecations = {k: (f"'brainscale.{k}' has been moved into brainscale.nn.{k}", getattr(nn, k)) for k in nn.__all__}
__getattr__ = deprecation_getattr(__name__, _deprecations)
del deprecation_getattr
