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

    .. tab-item:: GPU (CUDA 11.0)

       .. code-block:: bash

          pip install -U brainscale[cuda11]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainscale[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainscale[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


- `brainstate <https://github.com/brainpy/brainstate>`_: A ``State``-based transformation system for brain dynamics programming.

- `brainunit <https://github.com/brainpy/brainunit>`_: The unit system for brain dynamics programming.

- `braintaichi <https://github.com/brainpy/braintaichi>`_: Leveraging Taichi Lang to customize brain dynamics operators.

- `braintools <https://github.com/brainpy/braintools>`_: The toolbox for the brain dynamics simulation, training and analysis.

- `brainscale <https://github.com/brainpy/brainscale>`_: The scalable online learning framework for biological neural networks.



.. toctree::
   :hidden:
   :maxdepth: 2

   api.rst

