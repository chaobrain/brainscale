BrainScale documentation
========================

`BrainScale <https://github.com/brainpy/brainscale>`_ is designed for the scalable online learning of biological spiking neural networks.

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


We are building the `BDP ecosystem <https://ecosystem-for-brain-dynamics.readthedocs.io/>`_.

----



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Quickstart

   quickstart.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

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

   changelog.md
   api.rst

