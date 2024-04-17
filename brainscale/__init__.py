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

__version__ = "0.0.1"

from brainscale._base import *
from brainscale._base import __all__ as base_all
from brainscale._connections import *
from brainscale._connections import __all__ as connections_all
from brainscale._dynamics import *
from brainscale._dynamics import __all__ as dynamics_all
from brainscale._elementwise import *
from brainscale._elementwise import __all__ as elementwise_all
from brainscale._etrace_algorithms import *
from brainscale._etrace_algorithms import __all__ as etrace_algorithms
from brainscale._etrace_compiler import *
from brainscale._etrace_compiler import __all__ as etrace_compiler_all
from brainscale._etrace_concepts import *
from brainscale._etrace_concepts import __all__ as concepts_all
from brainscale._normalizations import *
from brainscale._normalizations import __all__ as normalizations_all
from brainscale._others import *
from brainscale._others import __all__ as others_all
from brainscale._poolings import *
from brainscale._poolings import __all__ as poolings_all
from brainscale._readout import *
from brainscale._readout import __all__ as readout_all
from brainscale._synouts import *
from brainscale._synouts import __all__ as synouts_all
from brainscale import typing

__all__ = (['typing'] +
           base_all +
           concepts_all +
           connections_all +
           dynamics_all +
           elementwise_all +
           etrace_algorithms +
           etrace_compiler_all +
           normalizations_all +
           others_all +
           poolings_all +
           readout_all +
           synouts_all)

del (base_all, concepts_all,
     connections_all, dynamics_all,
     elementwise_all, etrace_algorithms,
     normalizations_all, etrace_compiler_all,
     others_all,
     poolings_all, readout_all, synouts_all)
