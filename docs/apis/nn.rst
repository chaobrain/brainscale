``brainscale.nn`` for neural network building
=============================================

.. currentmodule:: brainscale.nn 
.. automodule:: brainscale.nn 



Connection Operation
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Linear
   ScaledWSLinear
   SignedWLinear
   CSRLinear
   Conv1d
   Conv2d
   Conv3d
   ScaledWSConv1d
   ScaledWSConv2d
   ScaledWSConv3d


Element-wise Operation
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ReLU
   RReLU
   Hardtanh
   ReLU6
   Sigmoid
   Hardsigmoid
   Tanh
   SiLU
   Mish
   Hardswish
   ELU
   CELU
   SELU
   GLU
   GELU
   Hardshrink
   LeakyReLU
   LogSigmoid
   Softplus
   Softshrink
   PReLU
   Softsign
   Tanhshrink
   Softmin
   Softmax
   Softmax2d
   LogSoftmax
   Dropout
   Dropout1d
   Dropout2d
   Dropout3d
   AlphaDropout
   FeatureAlphaDropout
   Identity
   SpikeBitwise


Neuronal Dynamics
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   IF
   LIF
   ALIF


Synaptic Dynamics
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Expon
   STP
   STD


Normalization Operation
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BatchNorm0d
   BatchNorm1d
   BatchNorm2d
   BatchNorm3d


Pooling Operation
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Flatten
   Unflatten
   AvgPool1d
   AvgPool2d
   AvgPool3d
   MaxPool1d
   MaxPool2d
   MaxPool3d
   AdaptiveAvgPool1d
   AdaptiveAvgPool2d
   AdaptiveAvgPool3d
   AdaptiveMaxPool1d
   AdaptiveMaxPool2d
   AdaptiveMaxPool3d


Rate RNNs
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ValinaRNNCell
   GRUCell
   MGUCell
   LSTMCell
   URLSTMCell
   MinimalRNNCell


Readout Layers
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LeakyRateReadout
   LeakySpikeReadout


