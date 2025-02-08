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

from pprint import pprint

import brainstate as bst
import brainunit as u
import pytest

import brainscale
from brainscale._etrace_compiler_base import (
    extract_model_info,
)
from brainscale._etrace_model_test import (
    IF_Delta_Dense_Layer,
    LIF_ExpCo_Dense_Layer,
    ALIF_ExpCo_Dense_Layer,
    LIF_ExpCu_Dense_Layer,
    LIF_STDExpCu_Dense_Layer,
    LIF_STPExpCu_Dense_Layer,
    ALIF_ExpCu_Dense_Layer,
    ALIF_Delta_Dense_Layer,
    ALIF_STDExpCu_Dense_Layer,
    ALIF_STPExpCu_Dense_Layer,
)


class Test_extract_model_info:
    def test_gru_one_layer(self):
        n_in = 3
        n_out = 4

        gru = brainscale.nn.GRUCell(n_in, n_out)
        bst.nn.init_all_states(gru)

        input = bst.random.rand(n_in)
        minfo = extract_model_info(gru, input)

    @pytest.mark.parametrize(
        'cls,',
        [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]
    )
    def test_snn_single_layer(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = cls(n_in, n_out)
            bst.nn.init_all_states(layer)
            minfo = extract_model_info(layer, input)
            pprint(minfo)

    @pytest.mark.parametrize(
        'cls,',
        [
            IF_Delta_Dense_Layer,
            LIF_ExpCo_Dense_Layer,
            ALIF_ExpCo_Dense_Layer,
            LIF_ExpCu_Dense_Layer,
            LIF_STDExpCu_Dense_Layer,
            LIF_STPExpCu_Dense_Layer,
            ALIF_ExpCu_Dense_Layer,
            ALIF_Delta_Dense_Layer,
            ALIF_STDExpCu_Dense_Layer,
            ALIF_STPExpCu_Dense_Layer,
        ]
    )
    def test_snn_two_layers(self, cls):
        n_in = 3
        n_out = 4
        input = bst.random.rand(n_in)

        print()
        print(cls)

        with bst.environ.context(dt=0.1 * u.ms):
            layer = bst.nn.Sequential(cls(n_in, n_out), cls(n_out, n_out))
            bst.nn.init_all_states(layer)
            minfo = extract_model_info(layer, input)
            pprint(minfo)

