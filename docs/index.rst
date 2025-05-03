``brainscalon`` documentation
============================

`brainscalon <https://github.com/chaobrain/brainscalon>`_ is designed for the scalable online learning of biological neural networks.

----



Basic Usage
^^^^^^^^^^^


Here we show how easy it is to use `brainscalon` to build and train a simple SNN/RNN model.



.. code-block::

   import brainscalon
   import brainstate as bst

   # define models as usual
   model = bst.nn.Sequential(
       bst.nn.GRU(2, 2),
       bst.nn.GRU(2, 1),
   )

   # initialize the model
   bst.nn.init_all_states(model)

   # the only thing you need to do just two lines of code
   model = brainscalon.DiagParamDimAlgorithm(model)
   model.compile_graph(your_inputs)

   # train your model as usual
   ...

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainscalon[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainscalon[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainscalon[tpu]


----


See also the brain modeling ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.

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

