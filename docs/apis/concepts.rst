Online Learning Concepts
========================

.. currentmodule:: brainscale

.. contents::
   :local:
   :depth: 1


ETrace Model Definition
-----------------------


If you are trying to define the hidden states for eligibility trace-based learning,
you can use the following classes to define the model.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceState
    ETraceGroupState



If you are trying to define the weight parameters for eligibility trace-based learning,
you can use the following classes to define the model.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceParam
    ETraceParamOp
    ElemWiseParamOp


If you do not want to compute weight gradients using eligibility trace-based learning,
you can use :py:class:`NonTempParamOp`, which computes the gradients using the standard
backpropagation algorithm at the current time step, while it is satisfying the
same interface as :py:class:`ETraceParamOp`.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    NonTempParamOp


If you do not want to compute weight gradients at all, you can use :py:class:`FakeParamOp`,
or :py:class:`FakeElemWiseParamOp`, which does not compute the gradients at all.


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    FakeParamOp
    FakeElemWiseParamOp



ETrace Operations
-----------------


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    ETraceOp
    StandardETraceOp
    GeneralETraceOp
    MatMulETraceOp


Others
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   stop_param_gradients

