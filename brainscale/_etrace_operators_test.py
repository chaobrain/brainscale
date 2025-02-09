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


import unittest

import jax.numpy as jnp

import brainscale
import brainstate as bst


class TestMatMulETraceOp(unittest.TestCase):
    def test_hidden_to_etrace_non_batch(self):
        w = {'weight': bst.random.rand(10, 20), 'bias': bst.random.rand(20)}
        dl_to_dh = bst.random.rand(20)
        dh_to_dw = {'weight': bst.random.randn(10, 20), 'bias': bst.random.randn(20)}

        etrace_op = brainscale.MatMulETraceOp()
        mode = bst.mixin.Mode()
        r = etrace_op.hidden_to_etrace(mode, w, dl_to_dh, dh_to_dw)

        self.assertTrue(
            jnp.allclose(
                r['weight'],
                dh_to_dw['weight'] * jnp.expand_dims(dl_to_dh, axis=0)
            )
        )
        self.assertTrue(
            jnp.allclose(
                r['bias'],
                dh_to_dw['bias'] * dl_to_dh
            )
        )

    def test_hidden_to_etrace_batch(self):
        n_batch = 2

        w = {'weight': bst.random.rand(10, 20), 'bias': bst.random.rand(20)}
        dl_to_dh = bst.random.rand(n_batch, 20)
        dh_to_dw = {'weight': bst.random.randn(n_batch, 10, 20), 'bias': bst.random.randn(n_batch, 20)}

        etrace_op = brainscale.MatMulETraceOp()
        mode = bst.mixin.Batching()
        r = etrace_op.hidden_to_etrace(mode, w, dl_to_dh, dh_to_dw)

        self.assertTrue(
            jnp.allclose(
                r['weight'],
                dh_to_dw['weight'] * jnp.expand_dims(dl_to_dh, axis=1)
            )
        )
        self.assertTrue(
            jnp.allclose(
                r['bias'],
                dh_to_dw['bias'] * dl_to_dh
            )
        )


class TestGeneralETraceOp(unittest.TestCase):
    def test_dx_dy_to_weight(self):
        pass
