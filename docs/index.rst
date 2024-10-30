``brainscale`` documentation
============================

`brainscale <https://github.com/chaobrain/brainscale>`_ is designed for the scalable online learning of biological spiking neural networks.

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainscale[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainscale[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainscale[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `Brain Dynamics Programming ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.

----



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/snn_online_learning-en.ipynb
   quickstart/rnn_online_learning-en.ipynb


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: 快速入门

   quickstart/concepts-zh.ipynb
   quickstart/snn_online_learning-zh.ipynb
   quickstart/rnn_online_learning-zh.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: 使用教程

   tutorials/etrace_concepts.ipynb
   tutorials/etrace_modules.ipynb
   tutorials/etrace_algorithms.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/core_examples.rst



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   api.rst

