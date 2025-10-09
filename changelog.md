# Release Notes


## Version 0.0.11

### Major Changes

#### Import Refactoring
- **Migrated imports from `brainstate` to `braintools`**: All initialization-related imports now use `braintools.init` instead of `brainstate.init`
  - Updated imports in:
    - `brainscale/nn/_neurons.py`: Changed `from brainstate import init` to `from braintools import init`
    - `brainscale/nn/_linear.py`: Changed `from brainstate import init` to `from braintools import init`
    - `brainscale/nn/_conv.py`: Updated initialization imports
    - `brainscale/nn/_synapses.py`: Updated initialization imports
    - `brainscale/nn/_readout.py`: Updated initialization imports

- **Migrated neural network model imports from `brainstate.nn` to `brainpy`**: Updated base classes for neuron models
  - `IF`, `LIF`, `ALIF` now inherit from `brainpy` instead of `brainstate.nn`
  - Maintained API compatibility while using the new `brainpy` backend

- **Updated functional API calls**: Changed from `brainstate.functional.sigmoid` to `brainstate.nn.sigmoid` in RNN cells

#### Dependency Updates
- **Added `brainpy` as a required dependency** in `requirements.txt`

#### Documentation Enhancements
- **Improved docstring formatting across the codebase**:
  - Enhanced parameter documentation with proper type annotations using NumPy-style docstrings
  - Added missing "Returns" sections to property and method docstrings
  - Converted inline examples to proper "Examples" sections with code blocks
  - Updated documentation in:
    - `brainscale/_etrace_algorithms.py`: Enhanced `EligibilityTrace` and `ETraceAlgorithm` documentation
    - `brainscale/_etrace_compiler_base.py`: Improved parameter and return type documentation
    - `brainscale/_etrace_compiler_module_info.py`: Enhanced module documentation

#### Core Algorithm Updates
- **RNN State Management**: Updated all RNN cells to use `braintools.init.param` for state initialization and reset
  - `ValinaRNNCell`: Updated `init_state()` and `reset_state()` methods
  - `GRUCell`: Updated state management and activation functions
  - `CFNCell`: Updated forget and input gate implementations
  - `MGUCell`: Updated minimal gated unit state handling

#### Test Updates
- **Refactored test imports**: Updated test files to use new import paths
  - `brainscale/_etrace_model_test.py`: Updated with new import structure
  - `brainscale/_etrace_vjp_algorithms_test.py`: Aligned with new API

#### Version
- Bumped version from `0.0.10` to `0.0.11`

### Files Changed (17 files)
- `.gitignore`: Added new patterns
- `brainscale/__init__.py`: Updated version number
- `brainscale/_etrace_algorithms.py`: Enhanced documentation and imports
- `brainscale/_etrace_compiler_base.py`: Improved documentation
- `brainscale/_etrace_compiler_graph.py`: Minor updates
- `brainscale/_etrace_compiler_hidden_group.py`: Minor updates
- `brainscale/_etrace_compiler_module_info.py`: Enhanced documentation
- `brainscale/_etrace_model_test.py`: Updated test imports
- `brainscale/_etrace_vjp_algorithms_test.py`: Updated test imports
- `brainscale/_etrace_vjp_graph_executor.py`: Updated imports
- `brainscale/nn/_conv.py`: Migrated to braintools imports
- `brainscale/nn/_linear.py`: Migrated to braintools imports
- `brainscale/nn/_neurons.py`: Migrated to brainpy and braintools
- `brainscale/nn/_rate_rnns.py`: Migrated to braintools and updated functional APIs
- `brainscale/nn/_readout.py`: Updated imports
- `brainscale/nn/_synapses.py`: Updated imports
- `requirements.txt`: Added brainpy dependency

### Breaking Changes
None. All changes maintain backward compatibility at the API level.

### Migration Guide
If you have custom code using brainscale:
- No changes required for end users
- If extending brainscale internally, note that initialization utilities now come from `braintools` instead of `brainstate`


