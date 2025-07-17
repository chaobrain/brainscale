``brainscale`` documentation
============================

`brainscale <https://github.com/chaobrain/brainscale>`_ is designed for the scalable online learning of biological neural networks.

----



Basic Usage
^^^^^^^^^^^


Here we show how easy it is to use `brainscale` to build and train a simple SNN/RNN model.



.. code-block::

   import brainscale
   import brainstate

   # define models as usual
   model = brainstate.nn.Sequential(
       brainstate.nn.GRU(2, 2),
       brainstate.nn.GRU(2, 1),
   )

   # initialize the model
   brainstate.nn.init_all_states(model)

   # the only thing you need to do just two lines of code
   model = brainscale.DiagParamDimAlgorithm(model)
   model.compile_graph(your_inputs)

   # train your model as usual
   ...

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


We are building the `brain modeling ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.


----


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb
   quickstart/snn_online_learning-en.ipynb
   quickstart/snn_online_learning-zh.ipynb
   quickstart/rnn_online_learning-en.ipynb
   quickstart/rnn_online_learning-zh.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Advanced Tutorial

   advanced/limitations-en.ipynb
   advanced/limitations-zh.ipynb




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
   apis/concepts.rst
   apis/compiler.rst
   apis/algorithms.rst
   apis/nn.rst

