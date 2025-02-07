Online Learning Compiler
========================

.. currentmodule:: brainscale


Base Classes
------------

The following classes are the base classes for the online learning compilation.

- :class:`HiddenGroup` summarizes all hidden state groups in the model.
  Each hidden state group contains multiple hidden state, and the hidden state transition function.
- :class:`HiddenParamOpRelation` summarizes the relation between hidden state groups,
  the associated parameter weights, and the operations that use them.
- :class:`CompiledGraph` contains the compiled graph of the model, including the
  hidden state groups, the operations, the parameter weights, the jaxpr of compiled models,
  and others.
- :class:`ETraceGraphExecutor` is used to implement or execute the compiled eligibility trace graph.
  It contains the compiled graph, the compiled model, the execution of the compiled model, and
  the interface to compute the Jacobian of hidden-group, and weight-to-hidden-group.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    HiddenGroup
    HiddenParamOpRelation
    CompiledGraph
    ETraceGraphExecutor


Compiler for VJP Algorithm
---------------------------

The following classes implement the compiler for the VJP-based online learning algorithms,
including those eligibility trace algorithms for:

- :class:`IODimVjpAlgorithm`
- :class:`ParamDimVjpAlgorithm`
- :class:`HybridDimVjpAlgorithm`


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    CompiledVjpGraph
    ETraceVjpGraphExecutor




