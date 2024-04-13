import functools
from enum import Enum, unique
from typing import Optional, Callable, Union

import braincore as bc
import jax
import jax.numpy as jnp
import numpy as np

from braintools import init
from ._etrace_concepts import ETraceVar, ETraceParam

__all__ = [
  'register_rtrl_method',

  'IdentityLayer',

  'ExpSmExactRTRL',
  'ExpSmDiagRTRL',
  'ExpSmLoraDiagRTRL',

  'TruncatedExactRTRL',
  'TruncatedDiagRTRL',
  'TruncatedLoraDiagRTRL',

  'OptLoraRTRL',
]

Array = Union[np.ndarray, jax.Array]

_default_algorithm = 'identity'
_registered_algorithms = {
  'identity': lambda x: x,
}


def register_rtrl_method(name: str, method: Callable, as_default: bool = False, override: bool = False):
  """Register the RTRL method.

  Args:
    name: str. The name of the RTRL method.
    method: Callable. The RTRL method.
    as_default: bool. Whether to set the method as the default method.
    override: bool. Whether to override the existing method.
  """
  if name in _registered_algorithms and not override:
    raise ValueError(f'The name {name} has been registered. ')
  _registered_algorithms[name] = method
  if as_default:
    global _default_algorithm
    _default_algorithm = name


def get_rtrl_method(method: Optional[Union[str, Callable]] = None):
  """Get the RTRL method.

  Args:
    method: str, callable. The RTRL method.

  Returns:
    The RTRL method.
  """
  if method is None:
    return _registered_algorithms[_default_algorithm]
  if isinstance(method, Callable):
    return method
  if isinstance(method, str):
    return _registered_algorithms[method]
  raise ValueError(f'Cannot find the RTRL method {method}.')


def _op_auto_grad(op, dx, w, dy):
  return jax.vjp(lambda weight: op(dx, weight), w)[1](dy)[0]


def _vmap_op_auto_grad(op, dx, w, dy, map_axis=1):
  f = lambda weight: (jax.vmap(op, in_axes=(map_axis, None), out_axes=map_axis)(dx, weight))
  return jax.vjp(f, w)[1](dy)[0]


def _check_shape(shape1, shape2):
  if shape1 != shape2:
    raise ValueError(f'The shape of the input is not consistent with the variable. '
                     f'Expected: {shape1}, Got: {shape2}.')


def _update_elg_trace(i, org_lefts, new_left, org_rights, new_right, axis=1):
  lshape = list(org_lefts.shape)
  num_rank = lshape.pop(axis)
  _check_shape(tuple(lshape), new_left.shape)

  rshape = list(org_rights.shape)
  assert rshape.pop(axis) == num_rank
  _check_shape(tuple(rshape), new_right.shape)

  sl = tuple([slice(None, None, None) for _ in range(axis)]) + (slice(1),)

  def _less_than():
    new_lefts = jnp.concatenate([org_lefts[sl], jnp.expand_dims(new_left, axis)], axis=axis)
    new_rights = jnp.concatenate([org_rights[sl], jnp.expand_dims(new_right, axis)], axis=axis)
    return new_lefts, new_rights

  def _bigger_than():
    new_lefts, new_rights = jax.vmap(_optimal_low_rank_approx)(
      jnp.concatenate([org_lefts, jnp.expand_dims(new_left, axis)], axis=axis),
      jnp.concatenate([org_rights, jnp.expand_dims(new_right, axis)], axis=axis)
    )
    return new_lefts, new_rights

  return jax.lax.cond(i < num_rank, _less_than, _bigger_than)


def _optimal_low_rank_approx(A_bar, B_bar):
  # A_bar [r+1, ...], B_bar [r+1, ...]
  A_shape = A_bar.shape[1:]
  B_shape = B_bar.shape[1:]
  assert A_bar.shape[0] == B_bar.shape[0]
  A_bar = bc.math.flatten(A_bar, start_dim=1).T  # [H, r+1]
  B_bar = bc.math.flatten(B_bar, start_dim=1).T  # [I, r+1]

  # QR factorization
  Q_A, R_A = jnp.linalg.qr(A_bar)  # Q_A [H, r+1]; R_A [r+1, r+1]
  Q_B, R_B = jnp.linalg.qr(B_bar)  # Q_B [I, r+1]; R_B [r+1, r+1]

  # SVD
  U, s, V = jnp.linalg.svd(R_A @ R_B.T)
  s_bar = jnp.sqrt(s[:-1])  # [r]
  r = s_bar.shape[0]
  U_bar = U[..., :-1] * s_bar  # [r+1, r]
  V_bar = V[:-1] * jnp.expand_dims(s_bar, 1)  # [r, r+1]

  # low-rank approximation
  A_new = Q_A @ U_bar  # [H, r]
  B_new = V_bar @ Q_B.T  # [r, I]
  A_new = jnp.reshape(A_new.T, (r,) + A_shape)  # [r, ...]
  B_new = jnp.reshape(B_new, (r,) + B_shape)  # [r, ...]
  return A_new, B_new


def _update_dict(the_dict, key, value):
  """Update the dictionary.

  If the key exists, then add the value to the existing value.
  Otherwise, create a new key-value pair.

  Args:
    the_dict: The dictionary.
    key: The key.
    value: The value.
  """
  if key in the_dict:
    the_dict[key] = the_dict[key] + value
  else:
    the_dict[key] = value


def _cosine_similarity(x, y):
  x = bc.math.flatten(x)
  y = bc.math.flatten(y)
  denominator = jnp.sqrt(jnp.sum(x * x)) * jnp.sqrt(jnp.sum(y * y))
  return jnp.where(denominator == 0, 0, jnp.sum(x * y) / denominator)


def compose(*funcs):
  """Compose the given functions.

  Args:
    *funcs: The functions.

  Returns:
    The composed function.
  """

  def _compose(f, g):
    return lambda x: f(g(x))

  return functools.reduce(_compose, funcs, lambda x: x)


def _norm(xs: dict):
  """Compute the L2 norm of the given variables.

  Args:
    xs: The dict variables.

  Returns:
    The L2 norm of the given variables.
  """
  r = 0.
  for x in xs.values():
    r += jnp.sum((x) ** 2, axis=tuple(range(x.ndim))[1:], keepdims=True)
  r = jnp.sqrt(r)
  return r


def _normalize(xs: dict):
  """Normalize the given variables.

  Args:
    xs: The dict variables.

  Returns:
    The normalized variables.
  """
  r = _norm(xs)
  res_xs = dict()
  for k in xs:
    res_xs[k] = xs[k] / r
  return res_xs


def _inner(a: jax.Array, b: jax.Array, keepdims=True):
  dim = tuple(range(a.ndim))
  assert a.shape == b.shape
  return jnp.sum(a * b, axis=dim[1:], keepdims=keepdims)


def _ones_at_given_pos(shape, position: int = 0, dtype=None):
  assert isinstance(shape, tuple)
  assert isinstance(position, int)
  ones = jnp.zeros(shape, dtype=dtype)
  ones[: position] = 1
  return ones


def _ones_like_at_given_pos(x, position: int = 0):
  return _ones_at_given_pos(x.shape, position, dtype=x.dtype)


def _expon_smooth(old, new, decay):
  return jax.tree_map(lambda x, y: decay * x + (1 - decay) * y, old, new)


def _low_pass_filter(old, new, alpha):
  return jax.tree_map(lambda x, y: alpha * x + y, old, new)


def _shift_with_batch(olds: Array, new: Array):
  # the first axis is the batch axis
  return jax.tree_map(jax.vmap(lambda x, y: jnp.concatenate([x[1:], jnp.expand_dims(y, 0)], axis=1)), olds, new)


def _format_decay_and_rank(decay, num_rank):
  # number of approximation rank
  # and the decay factor
  if num_rank is None:
    assert 0 < decay < 1, f'The decay should be in (0, 1). While we got {decay}. '
    decay = decay  # (num_rank - 1) / (num_rank + 1)
    num_rank = round(2. / (1 - decay) - 1)
  elif decay is None:
    assert isinstance(num_rank, int), f'The num_rank should be an integer. While we got {num_rank}. '
    num_rank = num_rank
    decay = (num_rank - 1) / (num_rank + 1)  # (num_rank - 1) / (num_rank + 1)
  else:
    raise ValueError('Please provide "num_rank" (int) or "decay" (float, 0 < decay < 1). ')
  return decay, num_rank


class NotSupportedError(Exception):
  pass


class _BaseEnum(Enum):

  @classmethod
  def get(cls, type_: Union[str, Enum]):
    if isinstance(type_, cls):
      return type_
    elif isinstance(type_, str):
      return cls.get_by_name(type_)
    else:
      raise ValueError(f'Cannot find the {cls.__name__} type {type_}.')

  @classmethod
  def get_by_name(cls, name: str):
    for item in cls:
      if item.name == name:
        return item
    raise ValueError(f'Cannot find the {cls.__name__} type {name}.')


@unique
class LORAApprox(_BaseEnum):
  """The type of the low-rank approximation.

  - `exact`: The exact low-rank approximation.
  - `diagonal`: The diagonal low-rank approximation.
  """
  jvp_rand = 'jvp_rand'
  vjp_rand = 'vjp_rand'
  vjp_rand_jvp = 'vjp_rand_jvp'
  vjp_onehot = 'vjp_onehot'


@unique
class JacobianApprox(_BaseEnum):
  """The type of the Jacobian matrix.

  - `jacobian`: The Jacobian matrix.
  - `diagonal`: The diagonal matrix of the Jacobian matrix.
  - `low_rank`: The low-rank approximation of the Jacobian matrix.
  """
  jacobian = 'jacobian'
  diagonal = 'diagonal'


class IdentityLayer(bc.Module):
  def __init__(self, layer):
    super().__init__()
    self.layer = layer

  def update(self, *args):
    return self.layer(*args)


class _RTRL(bc.Module):
  """The recurrent layer using the training algorithm of Real Time Recurrent Learning (RTRL).

  The keys for the memory-efficient computing are:

  1. Minimize the number of the eligibility trace variables. For example, does not create the
     spike variable.
  2. Minimize the number of the hidden variables.
  3. Use small batch size.
  4. The higher layer cannot use the variables in the lower layers. The lower layer can pass
     the variables to the higher layer using the "output_value()" function.

  """

  _excluded_vars = ('all_vars',)

  def __init__(
      self,
      layer: bc.Module,
      name: Optional[str] = None,
      mode: Optional[bc.mixin.Mode] = None
  ):
    super().__init__(name, mode)

    # the layer
    if not isinstance(layer, bc.Module):
      raise ValueError(f'The layer must be an instance of {bc.Module.__name__}. '
                       f'While we got {type(layer).__name__}')
    self.layer = layer

    # all variables
    self.all_vars = None

    # the function for the vjp gradient solving
    self.call_fun = jax.custom_vjp(self._fun_for_vjp)
    self.call_fun.defvjp(self._vjp_forward, self._vjp_backward)

  @bc.call_order(-1)
  def reset_state(self, *args, **kwargs):
    # Get all unique variables of the layer
    if self.all_vars is None:
      self.all_vars = self.layer.states().unique()
    all_vars = self.all_vars

    # Extract all variables of type ETVar from the unique variables
    elt_vars = all_vars.subset(ETraceVar)
    assert len(elt_vars) > 0, 'Found no eligibility trace variable. '

    # Extract all variables of type TrainVar from the unique variables
    train_vars = all_vars.subset(ETraceParam)
    assert len(train_vars) > 0, 'Found no eligibility trace training variable. '
    return elt_vars, train_vars

  @staticmethod
  def _elt_key(et_key, w_key):
    return f'{et_key} / {w_key}'

  def _fun_for_vjp(self, args: tuple, oth_vars: dict, elt_vars: dict, weights: dict):
    # assign the data
    for k, v in weights.items(): self.all_vars[k].value = v
    for k, v in elt_vars.items(): self.all_vars[k].value = v
    for k, v in oth_vars.items(): self.all_vars[k].value = v

    # update the layer
    out = self.layer(*args)

    # get new data
    new_elt_vars = self.all_vars.subset(ETraceVar).dict()
    new_oth_vars = self.all_vars.exclude(ETraceParam).exclude(ETraceVar).dict()
    return new_elt_vars, new_oth_vars

  def _vjp_forward(self, args, oth_vars, elt_vars, weights):
    """Customized VJP forward function.

    This function should be implemented by the subclass according to the specific RTRL algorithm.

    Args:
      args: The arguments for the layer.
      oth_vars: other variables defined in the layer.
      elt_vars: Eligibility trace variables defined in the layer.
      weights: Weight variables defined in the layer.

    Returns:
      The result of the forward function.
    """
    ret = self._fun_for_vjp(args, oth_vars, elt_vars, weights)
    raise NotImplementedError

    # elg_vars_inp = {k: v.value for k, v in self.elg_trace_inp.items()}
    # elg_vars_inp = jax.lax.stop_gradient(elg_vars_inp)
    # elg_vars_hid = {k: v.value for k, v in self.elg_trace_hid.items()}
    # elg_vars_hid = jax.lax.stop_gradient(elg_vars_hid)
    # return ret, ((elg_vars_inp, elg_vars_hid), (oth_vars, weights, elt_vars), args)

  def _vjp_backward(self, fwd_res, grads):
    """Customized VJP backward function.

    Args:
      fwd_res: The result of the forward function.
      grads: The gradients.

    Returns:
      The gradients for the input, the other variables, the weights, and the eligibility trace variables.
    """
    # necessary data
    dg_elt_vars, dg_oth_vars = grads
    elg_traces, variables, args = fwd_res
    all_vars_data = self.all_vars.dict()

    # calculate the original vector-valued gradients for the input
    dg_args = self._original_vjp_backward(dg_elt_vars, dg_oth_vars, elg_traces, variables, args)

    # calculate the vector-valued gradients for weights, combined with the eligibility trace
    dws = self._weight_backward(dg_elt_vars, elg_traces, variables, args)

    # recovery original data
    for k in all_vars_data: self.all_vars[k].value = all_vars_data[k]
    return dg_args, None, None, dws

  def _original_vjp_backward(self, dg_elt_vars, dg_oth_vars, elg_traces, variables, args):
    oth_vars, weights, elt_vars = variables
    f = functools.partial(self._fun_for_vjp, oth_vars=oth_vars, elt_vars=elt_vars, weights=weights)
    dargs = jax.vjp(f, args)[1]((dg_elt_vars, dg_oth_vars))[0]
    return dargs

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    raise NotImplementedError

  def _update_single_elt_var(self, elt_key, weights_data, et_vars_data, oth_vars_data, args):
    """Update the given eligibility trace variable.

    Args:
      elt_key: The key of the eligibility trace variable.
      weights_data: The weight variables.
      et_vars_data: The eligibility trace variables.
      oth_vars_data: The other variables.
      args: The arguments for the layer.

    Returns:
      The updated eligibility trace variable.
    """
    for k, v in weights_data.items(): self.all_vars[k].value = v
    for k, v in et_vars_data.items(): self.all_vars[k]._value = v
    for k, v in oth_vars_data.items(): self.all_vars[k]._value = v
    out = self.layer(*args)
    return self.all_vars[elt_key].value

  def _update_all_elt_vars(self, weights_data, et_vars_data, oth_vars_data, args):
    """Update all eligibility trace variables.

    Args:
      weights_data: The weight variables.
      et_vars_data: The eligibility trace variables.
      oth_vars_data: The other variables.
      args: The arguments for the layer.

    Returns:
      The updated eligibility trace variables.
    """
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    out = self.layer(*args)
    return {k: self.all_vars[k]._value for k in et_vars_data}

  def _update_state(self, *args):
    """Update the model states.

    Args:
      *args:
    """
    # get current data
    weights, elt_vars, oth_vars = self._get_variable_data()
    # update the model state
    new_elt_vars, new_oth_vars = self.call_fun(args, oth_vars, elt_vars, weights)
    # assign the updated states
    for k, v in new_elt_vars.items(): self.all_vars[k].value = v
    for k, v in new_oth_vars.items(): self.all_vars[k].value = v
    # recovery the weight
    for k, v in weights.items(): self.all_vars[k].value = v

  def _get_variable_data(self):
    # training weights
    weights_data = self.all_vars.subset(ETraceParam).dict()
    # eligibility trace variables
    et_vars_data = self.all_vars.subset(ETraceVar).dict()
    # other variables
    oth_vars_data = self.all_vars.exclude(ETraceVar).exclude(ETraceParam).dict()
    return weights_data, et_vars_data, oth_vars_data

  def _update_eligibility_trace(self, *args):
    """Update the eligibility trace.

    Args:
      *args:
    """
    raise NotImplementedError

  def update(self, *args):
    fit = bc.share.load('fit', desc='Indicating the fitting phase or not. ')

    # update the eligibility trace for later weight updates
    if fit and self.mode.is_train_mode():
      self._update_eligibility_trace(*args)

    # update the model states
    self._update_state(*args)

  def _assign_variable_data(self, *dicts):
    # recovery the dictionary data
    for a_dict in dicts:
      assert isinstance(a_dict, dict)
      for k, v in a_dict.items():
        self.all_vars[k].value = v

  def _assign_perturb_data(self, perturb_data):
    for k in perturb_data:
      self.all_vars[k].y_perturb = perturb_data[k]

  def _clear_perturb_data(self, weight_keys):
    for k in weight_keys:
      self.all_vars[k].y_perturb = None

  def _collect_x_primals(self, weight_keys):
    x_primals = dict()
    for ekey in weight_keys:
      x_primals[ekey] = self.all_vars[ekey].x_primal
    return x_primals

  def _record_x_primal(self, mode: bool):
    for var in self.all_vars.subset(ETraceParam).values():
      var.record_x_primal = mode

  def _stop_weight_gradients(self, mode: bool):
    for var in self.all_vars.subset(ETraceParam).values():
      var._stop_weight_gradient = mode


class _RankNRTRL(_RTRL):
  """Real-Time Recurrent Learning with Rank-n Approximation.
  """

  def _get_hidden_etrace_of_given_weight(self, w_key):
    res = dict()
    elt_vars = self.all_vars.subset(ETraceVar).dict()
    for elt_key, elt_data in elt_vars.items():
      elt_shape = elt_data.shape

      ew_key = self._elt_key(elt_key, w_key)
      if ew_key in self.elg_trace_hid:
        hid_trace = self.elg_trace_hid[ew_key].value
      else:
        hid_trace = jnp.zeros((elt_shape[0], self.num_rank) + elt_shape[1:])
      res[elt_key] = hid_trace
    return res


class _RTRLWithDiag(_RTRL):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # The hidden to hidden Jacobian transition
    self.elg_trace_hid = bc.visible_state_dict()

    # The input to hidden Jacobian transition
    self.elg_trace_inp = bc.visible_state_dict()

  def _vjp_forward(self, args, oth_vars, elt_vars, weights):
    """Customized VJP forward function.

    This function should be implemented by the subclass according to the specific RTRL algorithm.

    Args:
      args: The arguments for the layer.
      oth_vars: other variables defined in the layer.
      elt_vars: Eligibility trace variables defined in the layer.
      weights: Weight variables defined in the layer.

    Returns:
      The result of the forward function.
    """
    ret = self._fun_for_vjp(args, oth_vars, elt_vars, weights)

    elg_vars_inp = {k: v.value for k, v in self.elg_trace_inp.items()}
    elg_vars_inp = jax.lax.stop_gradient(elg_vars_inp)
    elg_vars_hid = {k: v.value for k, v in self.elg_trace_hid.items()}
    elg_vars_hid = jax.lax.stop_gradient(elg_vars_hid)
    return ret, ((elg_vars_inp, elg_vars_hid), (oth_vars, weights, elt_vars), args)

  def _update_eligibility_trace(self, *args):
    raise NotImplementedError


class _RTRLWithLoraAndInpHid(_RTRL):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # eligibility trace
    self.elg_trace_lora_lefts = bc.visible_state_dict()
    self.elg_trace_lora_rights = bc.visible_state_dict()
    self.elg_trace_lora_inp = bc.visible_state_dict()
    self.elg_trace_inp = bc.visible_state_dict()
    self.elg_trace_hid = bc.visible_state_dict()
    self.elg_trace_hid_prev = bc.visible_state_dict()
    self.elg_trace_inp_prev = bc.visible_state_dict()

  def _vjp_forward(self, args, oth_vars, elt_vars, weights):
    ret = self._fun_for_vjp(args, oth_vars, elt_vars, weights)

    # the diagonal approximation
    elg_trace_inp = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_inp.items()})
    elg_trace_hid = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_hid.items()})

    # the low-rank approximation
    elg_trace_lora_inp = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_lora_inp.items()})
    elg_trace_lora_rights = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_lora_rights.items()})
    elg_trace_lora_lefts = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_lora_lefts.items()})
    return ret, ((elg_trace_inp, elg_trace_hid, elg_trace_lora_inp, elg_trace_lora_lefts, elg_trace_lora_rights),
                 (oth_vars, weights, elt_vars),
                 args)

  def _update_eligibility_trace(self, *args):
    raise NotImplementedError


class _RTRLWithLoraDiag(_RTRL):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # eligibility trace
    self.elg_trace_lora_lefts = bc.visible_state_dict()
    self.elg_trace_lora_rights = bc.visible_state_dict()
    self.elg_trace_lora_inp = bc.visible_state_dict()
    self.elg_trace_hid = bc.visible_state_dict()
    self.elg_trace_inp = bc.visible_state_dict()

  def _vjp_forward(self, args, oth_vars, elt_vars, weights):
    ret = self._fun_for_vjp(args, oth_vars, elt_vars, weights)

    # the diagonal approximation
    elg_trace_inp = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_inp.items()})
    elg_trace_hid = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_hid.items()})

    # the low-rank approximation
    elg_trace_lora_inp = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_lora_inp.items()})
    elg_trace_lora_rights = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_lora_rights.items()})
    elg_trace_lora_lefts = jax.lax.stop_gradient({k: v.value for k, v in self.elg_trace_lora_lefts.items()})
    return ret, ((elg_trace_inp, elg_trace_hid, elg_trace_lora_inp, elg_trace_lora_lefts, elg_trace_lora_rights),
                 (oth_vars, weights, elt_vars),
                 args)

  def _update_eligibility_trace(self, *args):
    raise NotImplementedError


class ExpSmExactRTRL(_RTRLWithDiag):
  """Real-Time Recurrent Learning (RTRL) with the exponential decay trace of the input and hidden primal data.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).

  - The model has O(n^2) memory complexity.
  - It can only be used to train linear transformation layers.

  Args:
    layer: The layer to be trained.
    decay: The decay factor.
    num_rank: The number of the approximation rank.
    name: The name of the model.
    mode: The mode of the model.

  """

  def __init__(
      self,
      layer: bc.Module,
      decay: float = None,
      num_rank: int = None,
      **kwargs
  ):
    super().__init__(layer, **kwargs)

    self.decay, self.num_rank = _format_decay_and_rank(decay, num_rank)

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the Jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            if len(w_var.yshape) != 1:
              raise NotSupportedError(f'The {self.__class__.__name__} method only '
                                      'supports the linear transformation layer. ')
            key = self._elt_key(et_key, w_key)
            # the O(n^2) memory complexity
            self.elg_trace_hid[key] = init.state(jnp.zeros, (batch_size,) + w_var.yshape + w_var.yshape)
            count += 1
        # input variable
        if count > 0:
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, (batch_size,) + w_var.xshape)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dws = dict()
    ew_vars_inp, ew_vars_hid = elg_trace_vars
    _, weights, elt_vars = variables
    for wkey in weights:
      w = weights[wkey]  # weight
      dx = ew_vars_inp[wkey]  # input eligibility trace
      op = self.all_vars[wkey].op  # operator of the weight

      for ekey in elt_vars.keys():
        key = self._elt_key(ekey, wkey)  # eligibility trace key
        if key in ew_vars_hid:
          # the hidden eligibility trace
          dy = jax.vmap(jnp.matmul)(dg_elt_vars[ekey], ew_vars_hid[key])  # [B, H, H]
          _update_dict(dws, wkey, _op_auto_grad(op, dx, w, dy))
    dws = {k: (v) for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, *args):
    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    def _get_hidden_etrace_of_given_weight(w_key):
      res = dict()
      for e_key, elt_data in et_vars_data.items():
        ew_key = self._elt_key(e_key, w_key)
        hid_trace = (
          self.elg_trace_hid[ew_key].value  # [B, H, H]
          if ew_key in self.elg_trace_hid else
          jnp.zeros(elt_data.shape + elt_data.shape[1:])  # [B, H, H]
        )
        res[e_key] = hid_trace
      return res

    # the function returns the given eligibility trace variable
    f_for_jac = lambda et_vars: self._update_all_elt_vars(weights_data, et_vars, oth_vars_data, args)

    # ----------------------- #
    # step 1: jacobian update #
    # ----------------------- #

    for w_key in weights_data:
      # the weight gradients of the given eligibility trace variable
      eweights = _get_hidden_etrace_of_given_weight(w_key)

      # jacobian vector product #
      # ----------------------- #
      # Therefore, using the "jax.jvp" function is enough.
      jvp_res = jax.vmap(lambda ews: jax.jvp(f_for_jac, (et_vars_data,), (ews,))[1], in_axes=2, out_axes=2)(eweights)

      # the hidden to hidden Jacobian transition
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_hid:
          self.elg_trace_hid[key] = jvp_res[elt_key]

      # recovery data
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # ------------------------------- #
    # step 2: new jacobian and inputs #
    # ------------------------------- #

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # perturb data
    bs = args[0].shape[0]
    perturb_data = {k: jnp.zeros((bs,) + v.yshape) for k, v in self.all_vars.subset(ETraceParam).items()}
    # vector-valued gradients computing the right cotangents
    yy, f_vjp, inputs = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # [KEY]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        self.elg_trace_inp[w_key] = _low_pass_filter(self.elg_trace_inp[w_key], inputs[w_key], self.decay)

    # [KEY]
    # update the right cotangents
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      right_cotangents = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          self.elg_trace_hid[ew_key] = _expon_smooth(self.elg_trace_hid[ew_key],
                                                     jax.vmap(jnp.diag)(right_cotangents[w_key]),
                                                     self.decay)

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())


class ExpSmDiagRTRL(_RTRLWithDiag):
  """Real-Time Recurrent Learning (RTRL) with the exponential decay trace of the input and hidden primal data.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).

  Args:
    layer: The layer to be trained.
    decay: The decay factor.
    num_rank: The number of the approximation rank.
    name: The name of the model.
    mode: The mode of the model.

  """

  def __init__(
      self,
      layer: bc.Module,
      decay: float = None,
      num_rank: int = None,
      **kwargs
  ):
    super().__init__(layer, **kwargs)

    self.decay, self.num_rank = _format_decay_and_rank(decay, num_rank)

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the Jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            key = self._elt_key(et_key, w_key)
            self.elg_trace_hid[key] = init.state(jnp.zeros, (batch_size,) + w_var.yshape)
            count += 1

        # input variable
        if count > 0:
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, (batch_size,) + w_var.xshape)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dws = dict()
    elg_keys = list(self.all_vars.subset(ETraceVar).keys())
    _, weights, _ = variables
    ew_vars_inp, ew_vars_hid = elg_trace_vars

    for wkey in weights:
      w = weights[wkey]  # weight
      dx = ew_vars_inp[wkey]  # input eligibility trace
      op = self.all_vars[wkey].op  # operator of the weight
      for ekey in elg_keys:
        key = self._elt_key(ekey, wkey)
        if key in ew_vars_hid:
          # [KEY]
          # Here, we also have an issue that the hidden eligibility trace ``ew_vars_hid[key]``
          # may have a different shape with the hidden state gradients ``dg_elt_vars[ekey]``.
          dy = ew_vars_hid[key] * dg_elt_vars[ekey]  # hidden eligibility trace
          _update_dict(dws, wkey, _op_auto_grad(op, dx, w, dy))
    dws = {k: v for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, *args):
    """Separate the hidden to weight Jacobian into two parts: left and right cotangents.

    This function is the key, since the separation of left and right
    cotangents is the key for the low-rank approximation.

    Args:
      args: The arguments for the layer.
    """
    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    # ----------------------- #
    # step 1: jacobian update #
    # ----------------------- #

    # the function returns the given eligibility trace variable
    f_for_jac = lambda et_vars: self._update_all_elt_vars(weights_data, et_vars, oth_vars_data, args)
    # [compute D^t]
    # approximate the hidden to hidden Jacobian diagonal using the JVP
    self._stop_weight_gradients(True)
    diagonal = bm.vector_grad(f_for_jac, argnums=0)(et_vars_data)
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)  # recovery data
    self._stop_weight_gradients(False)

    # update the hidden to hidden Jacobian
    for w_key in weights_data:
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_hid:
          # [KEY]
          # This operation requires that the hidden state should be the same as the operator output.
          # For generalization, we need define a mixing function to specify how to mix the hidden state
          # and the eligibility trace (the operator output).
          # ----
          # Actually, this term is involved in how to mix
          # the current diagonal with the previous diagonal eligibility trace.
          self.elg_trace_hid[key] = diagonal[elt_key] * self.elg_trace_hid[key]

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # ------------------------------- #
    # step 2: new jacobian and inputs #
    # ------------------------------- #

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # perturb data
    bs = args[0].shape[0]
    perturb_data = {k: jnp.zeros((bs,) + v.yshape, dtype=v.dtype)
                    for k, v in self.all_vars.subset(ETraceParam).items()}
    # [Compute x^t]
    # vector-valued gradients computing the right cotangents
    yy, f_vjp, inputs = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # [KEY]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        self.elg_trace_inp[w_key] = _low_pass_filter(self.elg_trace_inp[w_key],
                                                     inputs[w_key],
                                                     self.decay)

    # [KEY]
    # update the right cotangents
    for elt_key in et_vars_data:
      # [Compute D_f^t]
      # There is no weight-transpose, since the required derivative are only about the hidden states.
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      dfs = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          self.elg_trace_hid[ew_key] = _expon_smooth(self.elg_trace_hid[ew_key],
                                                     dfs[w_key],
                                                     self.decay)

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())


class ExpSmLoraDiagRTRL(_RTRLWithLoraDiag):
  """Real-Time Recurrent Learning (RTRL) with the exponential decay trace of the input and hidden primal data.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).

  - Low-rank approximation is used for the hidden to hidden Jacobian transition.
  - Exponential smoothing is used for the Jocobian composition of multiple time steps.

  """

  def __init__(
      self,
      layer: bc.Module,
      decay: Optional[float] = None,
      num_rank: Optional[int] = None,
      lora_approx: Union[LORAApprox, str] = LORAApprox.vjp_rand,
      **kwargs
  ):
    super().__init__(layer, **kwargs)

    # LORA approximation
    self.lora_approx = LORAApprox.get(lora_approx)

    self.decay, self.num_rank = _format_decay_and_rank(decay, num_rank)

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        # batch size

        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            key = self._elt_key(et_key, w_key)
            # current hidden jacobian f'
            self.elg_trace_hid[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            # LORA hidden lefts
            self.elg_trace_lora_lefts[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            # LORA hidden rights
            self.elg_trace_lora_rights[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            count += 1

        # input variable
        if count > 0:
          # current input
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, w_var.xshape, batch_size)
          # LORA inputs
          self.elg_trace_lora_inp[w_key] = init.state(jnp.zeros, w_var.xshape, batch_size)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dws = dict()
    elg_keys = tuple(self.all_vars.subset(ETraceVar).keys())
    elg_trace_inp, elg_trace_hid, elg_trace_lora_inp, elg_trace_lora_lefts, elg_trace_lora_rights = elg_trace_vars
    _, weights, _ = variables
    for w_key in weights:
      w = weights[w_key]  # weight
      op = self.all_vars[w_key].op  # operator of the weight
      dx = elg_trace_inp[w_key]  # input eligibility trace of LORA
      dx_lora = elg_trace_lora_inp[w_key]  # input eligibility trace of LORA

      for ekey in elg_keys:
        key = self._elt_key(ekey, w_key)
        if key in elg_trace_hid:
          # the gradients computed by the low-ranked approximation
          dy_lora = (
              _inner((dg_elt_vars[ekey]),  # loss
                     (elg_trace_lora_lefts[key]),  # LORA lefts
                     keepdims=True)
              * (elg_trace_lora_rights[key])  # LORA rights
          )
          _update_dict(dws, w_key, _op_auto_grad(op, dx_lora, w, dy_lora))

          # the gradients computed by the diagonal approximation
          dy = (dg_elt_vars[ekey]) * elg_trace_hid[key]
          _update_dict(dws, w_key, _op_auto_grad(op, dx, w, dy))

    dws = {k: (v) for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, *args):
    """Computing the hidden-to-hidden Jacobian transition.

    Args:
      args: tuple. The arguments for the layer.
    """
    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    # get the hidden to weight Jacobian #
    # --------------------------------- #

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # Step 1: LORA_e = decay * LORA_e + b_prev
    for w_key in weights_data:
      if w_key in self.elg_trace_lora_inp:
        self.elg_trace_lora_inp[w_key] = _low_pass_filter(self.elg_trace_lora_inp[w_key],
                                                          self.elg_trace_inp[w_key],
                                                          self.decay)

    # get the batch size
    bs = args[0].shape[0]

    # generate perturbation data
    perturb_data = {k: jnp.zeros((bs,) + v.yshape, dtype=v.dtype)
                    for k, v in self.all_vars.subset(ETraceParam).items()}
    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # vector-valued gradients computing the right cotangents #
    yy, f_vjp, inputs = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # Step 2: record the "x" in [ \partial h / \partial W = f' \otimes x ]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        self.elg_trace_inp[w_key] = _low_pass_filter(self.elg_trace_inp[w_key],
                                                     inputs[w_key],
                                                     self.decay)

    # Step 3: record the "f'" in [ \partial h / \partial W = f' \otimes x ]
    current_h2w_diag = {}
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      dfs = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          current_h2w_diag[ew_key] = dfs[w_key]

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())

    # update the hidden to hidden Jacobian #
    # ------------------------------------ #

    f_for_jac = lambda et_vars: self._update_all_elt_vars(weights_data, et_vars, oth_vars_data, args)

    # Step 1: approximate the hidden to hidden diagonal Jacobian #
    self._stop_weight_gradients(True)
    h2h_diagonal = bm.vector_grad(f_for_jac, argnums=0)(et_vars_data)
    self._stop_weight_gradients(False)
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)

    # Step 2: approximate the maximum eigenvalues and eigenvectors of (Jacobian - Diagonal) #
    # [A]. Compute the dominant eigenvector using power iteration, only using one interation
    #      (Jacobian - Diagonal) @ x = (Jacobian @ x) - (Diagonal @ x)
    if self.lora_approx == LORAApprox.vjp_onehot:
      rights = jax.tree_map(_ones_like_at_given_pos, et_vars_data)
      rights = jax.vjp(f_for_jac, et_vars_data)[1](rights)[0]
      rights = _normalize(rights)
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
      lefts = jax.jvp(f_for_jac, (et_vars_data,), (rights,))[1]
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
      lefts = jax.tree_map(lambda a, d, c: a - d * c, lefts, h2h_diagonal, rights)

    elif self.lora_approx == LORAApprox.vjp_rand:
      lefts = jax.tree_map(bc.random.randn_like, et_vars_data)
      lefts = _normalize(lefts)
      rights = jax.vjp(f_for_jac, et_vars_data)[1](lefts)[0]
      rights = jax.tree_map(lambda a, d, c: a - d * c, rights, h2h_diagonal, lefts)
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)

    elif self.lora_approx == LORAApprox.vjp_rand_jvp:
      lefts = jax.tree_map(bc.random.randn_like, et_vars_data)
      lefts = _normalize(lefts)
      rights = jax.vjp(f_for_jac, et_vars_data)[1](lefts)[0]
      rights = jax.tree_map(lambda a, d, c: a - d * c, rights, h2h_diagonal, lefts)
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
      rights = _normalize(rights)
      lefts = jax.jvp(f_for_jac, (et_vars_data,), (rights,))[1]
      lefts = jax.tree_map(lambda a, d, c: a - d * c, lefts, h2h_diagonal, rights)
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)

    elif self.lora_approx == LORAApprox.jvp_rand:
      lefts = jax.tree_map(bc.random.rand_like, et_vars_data)
      lefts = _normalize(lefts)
      rights = jax.jvp(f_for_jac, (et_vars_data,), (lefts,))[1]
      rights = jax.tree_map(lambda a, d, c: a - d * c, rights, h2h_diagonal, lefts)

    else:
      raise ValueError(f'Please provide a valid LORA approximation. See {LORAApprox}')

    # [B]. Compute the dominant eigenvalue using Rayleigh quotient
    # eigen_value = 0.
    # for elt_key in et_vars_data:
    #   eigen_value += jnp.sum(rights[elt_key] * lefts[elt_key], keepdims=True,
    #                          axis=tuple(range(rights[elt_key].ndim))[1:])

    # [C]. Compute the rank-one approximation of the Jacobian
    #      using the dominant eigenvalue and eigenvector:
    #      J = eigenval * iter_init \otimes iter_next

    # Step 3: update the hidden to hidden Jacobian #

    def get_lora_lefts(elt_key, w_key):
      k_ = self._elt_key(elt_key, w_key)
      if k_ in self.elg_trace_lora_lefts:
        return self.elg_trace_lora_lefts[k_].value
      else:
        return jnp.zeros_like(et_vars_data[elt_key])

    for w_key in weights_data:
      # [A]. Jacobian @ LORA: using JVP
      #      Jacobian @ LORA = Jacobian @ LORA_left @ LORA_right
      #                    = new_left @ LORA_right
      etweights = {elt_key: get_lora_lefts(elt_key, w_key) for elt_key in et_vars_data}
      J_lefts = jax.jvp(f_for_jac, (et_vars_data,), (etweights,))[1]
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)  # recovery data

      # [B]. update low-rank approximation
      #      LORA_left = decay * new_left + (1 - decay) * current_LORA_left
      #      LORA_right = decay * old_LORA_right + (1 - decay) * new_LORA_right
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_lora_lefts:
          self.elg_trace_lora_lefts[key] = _expon_smooth(J_lefts[elt_key],  # old LEFT
                                                         lefts[elt_key],  # current LEFT
                                                         self.decay)
          self.elg_trace_lora_rights[key] = _expon_smooth(self.elg_trace_lora_rights[key],  # old RIGHT
                                                          rights[elt_key] *
                                                          self.elg_trace_hid[key],  # current RIGHT
                                                          self.decay)

      # [C]. update diagonal approximation #
      #      Diagonal = decay * old_Diagonal + (1 - decay) * new_Diagonal
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_hid:
          self.elg_trace_hid[key] = _expon_smooth(h2h_diagonal[elt_key] * self.elg_trace_hid[key],
                                                  current_h2w_diag[key],
                                                  self.decay)


class ExponSmoothLoraDiagRTRL_v2(_RTRLWithLoraAndInpHid):
  """Real-Time Recurrent Learning (RTRL) with the exponential decay trace of the input and hidden primal data.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).

  - Low-rank approximation is used for the hidden to hidden Jacobian transition.
  - Exponential smoothing is used for the Jocobian composition of multiple time steps.

  """

  def __init__(
      self,
      layer: bc.Module,
      decay: Optional[float] = None,
      num_rank: Optional[int] = None,
      lora_approx: Union[LORAApprox, str] = LORAApprox.vjp_onehot,
      **kwargs
  ):
    super().__init__(layer, **kwargs)

    # LORA approximation
    self.lora_approx = LORAApprox.get(lora_approx)

    self.decay, self.num_rank = _format_decay_and_rank(decay, num_rank)

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        # batch size

        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            key = self._elt_key(et_key, w_key)
            # current hidden jacobian f'
            self.elg_trace_hid[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            # previous hidden jacobian f'
            self.elg_trace_hid_prev[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            # LORA hidden lefts
            self.elg_trace_lora_lefts[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            # LORA hidden rights
            self.elg_trace_lora_rights[key] = init.state(jnp.zeros, tuple(w_var.yshape), batch_size)
            count += 1

        # input variable
        if count > 0:
          # current input
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, w_var.xshape, batch_size)
          # previous input
          self.elg_trace_inp_prev[w_key] = init.state(jnp.zeros, w_var.xshape, batch_size)
          # LORA inputs
          self.elg_trace_lora_inp[w_key] = init.state(jnp.zeros, w_var.xshape, batch_size)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dws = dict()
    elg_keys = tuple(self.all_vars.subset(ETraceVar).keys())
    elg_trace_inp, elg_trace_hid, elg_trace_lora_inp, elg_trace_lora_lefts, elg_trace_lora_rights = elg_trace_vars
    _, weights, _ = variables
    for w_key in weights:
      w = weights[w_key]  # weight
      op = self.all_vars[w_key].op  # operator of the weight
      dx = elg_trace_inp[w_key]  # input eligibility trace of LORA
      dx_lora = elg_trace_lora_inp[w_key]  # input eligibility trace of LORA

      for ekey in elg_keys:
        key = self._elt_key(ekey, w_key)
        if key in elg_trace_hid:
          # the gradients computed by the low-ranked approximation
          dy_lora = (
              _inner((dg_elt_vars[ekey]),  # loss
                     (elg_trace_lora_lefts[key]),  # LORA lefts
                     keepdims=True)
              * (elg_trace_lora_rights[key])  # LORA rights
          )
          _update_dict(dws, w_key, _op_auto_grad(op, dx_lora, w, dy_lora))

          # the gradients computed by the diagonal approximation
          dy = (dg_elt_vars[ekey]) * elg_trace_hid[key]
          _update_dict(dws, w_key, _op_auto_grad(op, dx, w, dy))

    dws = {k: (v) for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, *args):
    """Computing the hidden-to-hidden Jacobian transition.

    Args:
      args: tuple. The arguments for the layer.
    """
    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    # get the hidden to weight Jacobian #
    # --------------------------------- #

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # get the batch size
    bs = args[0].shape[0]

    # generate perturbation data
    perturb_data = {k: jnp.zeros((bs,) + v.yshape) for k, v in self.all_vars.subset(ETraceParam).items()}

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # vector-valued gradients computing the right cotangents #
    yy, f_vjp, inputs = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # Step 1: record the "x" in [ \partial h / \partial W = f' \otimes x ]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        self.elg_trace_inp[w_key] = _low_pass_filter(self.elg_trace_inp[w_key], inputs[w_key], self.decay)

    # Step 2: record the "f'" in [ \partial h / \partial W = f' \otimes x ]
    current_h2w_diag = {}
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      dfs = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          current_h2w_diag[ew_key] = dfs[w_key]

    # Step 3: LORA_x = decay * LORA_x + (1 - decay) * current_x
    for w_key in weights_data:
      if w_key in self.elg_trace_lora_inp:
        self.elg_trace_lora_inp[w_key] = _low_pass_filter(self.elg_trace_lora_inp[w_key],
                                                          self.elg_trace_inp_prev[w_key],
                                                          self.decay)
        # the current input is the previous input for Jacobian multiplication of the next time step
        self.elg_trace_inp_prev[w_key] = inputs[w_key]

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())

    # update the hidden to hidden Jacobian #
    # ------------------------------------ #

    f_for_jac = lambda et_vars: self._update_all_elt_vars(weights_data, et_vars, oth_vars_data, args)

    # Step 1: approximate the hidden to hidden diagonal Jacobian #
    self._stop_weight_gradients(True)
    diag_approx = bm.vector_grad(f_for_jac, argnums=0)(et_vars_data)
    # _init_grads = jax.tree_map(jnp.ones_like, et_vars_data)
    # diag_approx = jax.jvp(f_for_jac, (et_vars_data,), (_init_grads,))[1]
    self._stop_weight_gradients(False)
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)

    # Step 2: approximate the maximum eigenvalues and eigenvectors of (Jacobian - Diagonal) #
    # [A]. Compute the dominant eigenvector using power iteration, only using one interation
    #      (Jacobian - Diagonal) @ x = (Jacobian @ x) - (Diagonal @ x)
    if self.lora_approx == LORAApprox.vjp_onehot:
      rights = jax.tree_map(_ones_like_at_given_pos, et_vars_data)
      rights = jax.vjp(f_for_jac, et_vars_data)[1](rights)
      rights = _normalize(rights)
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
      lefts = jax.jvp(f_for_jac, (et_vars_data,), (rights,))[1]
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
      lefts = jax.tree_map(lambda a, b, c: a - b * c, lefts, diag_approx, rights)
    elif self.lora_approx == LORAApprox.vjp_rand:
      lefts = jax.tree_map(_ones_like_at_given_pos, et_vars_data)
      lefts = _normalize(lefts)
      rights = jax.vjp(f_for_jac, et_vars_data)[1](lefts)
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
      rights = jax.tree_map(lambda a, b, c: a - b * c, rights, diag_approx, lefts)
    elif self.lora_approx == LORAApprox.jvp_rand:
      lefts = jax.tree_map(lambda x: bc.random.rand_like(x), et_vars_data)
      lefts = _normalize(lefts)
      rights = jax.jvp(f_for_jac, (et_vars_data,), (lefts,))[1]
      rights = jax.tree_map(lambda a, b, c: a - b * c, rights, diag_approx, lefts)
    else:
      raise ValueError(f'Please provide a valid LORA approximation. See {LORAApprox}')

    # [B]. Compute the dominant eigenvalue using Rayleigh quotient
    # eigen_value = 0.
    # for elt_key in et_vars_data:
    #   eigen_value += jnp.sum(rights[elt_key] * lefts[elt_key], keepdims=True,
    #                          axis=tuple(range(rights[elt_key].ndim))[1:])

    # [C]. Compute the rank-one approximation of the Jacobian
    #      using the dominant eigenvalue and eigenvector:
    #      J = eigenval * iter_init \otimes iter_next

    # Step 3: update the hidden to hidden Jacobian #

    def get_lora_lefts(elt_key, w_key):
      key = self._elt_key(elt_key, w_key)
      if key in self.elg_trace_lora_lefts:
        return self.elg_trace_lora_lefts[key].value
      else:
        return jnp.zeros_like(et_vars_data[elt_key])

    for w_key in weights_data:
      # [A]. Jacobian @ LORA: using JVP
      #      Jacobian @ LORA = Jacobian @ LORA_left @ LORA_right
      #                    = new_left @ LORA_right
      etweights = {elt_key: get_lora_lefts(elt_key, w_key) for elt_key in et_vars_data}
      new_lefts = jax.jvp(f_for_jac, (et_vars_data,), (etweights,))[1]
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)  # recovery data

      # [B]. update low-rank approximation
      #      LORA_left = decay * new_left + (1 - decay) * (LORA_left - diagonal * LORA_right)
      #      LORA_right = decay * old_LORA_right + (1 - decay) * new_LORA_right
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_lora_lefts:
          self.elg_trace_lora_lefts[key] = _expon_smooth(new_lefts[elt_key],  # old LEFT
                                                         # eigen_value * lefts[elt_key],  # current LEFT
                                                         lefts[elt_key],  # current LEFT
                                                         self.decay)
          self.elg_trace_lora_rights[key] = _expon_smooth(self.elg_trace_lora_rights[key],  # old RIGHT
                                                          rights[elt_key] *
                                                          self.elg_trace_hid_prev[key],  # current RIGHT
                                                          self.decay)
          self.elg_trace_hid_prev[key] = current_h2w_diag[key]

      # [C]. update diagonal approximation #
      #      Diagonal = decay * old_Diagonal + (1 - decay) * new_Diagonal
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_hid:
          self.elg_trace_hid[key] = _expon_smooth(diag_approx[elt_key] * self.elg_trace_hid[key],
                                                  current_h2w_diag[key],
                                                  self.decay)


class TruncatedExactRTRL(_RTRLWithDiag):
  def __init__(
      self,
      layer: bc.Module,
      num_rank: int,
      **kwargs
  ):
    super().__init__(layer, **kwargs)

    # number of approximation rank
    self.num_rank = num_rank

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            if len(w_var.yshape) != 1:
              raise NotSupportedError(f'The {self.__class__.__name__} method only '
                                      f'supports the linear transformation layer. ')
            key = self._elt_key(et_key, w_key)
            # the O(rn^2) memory complexity
            self.elg_trace_hid[key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.yshape + w_var.yshape),
                                                batch_size)
            count += 1

        # input variable
        if count > 0:
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.xshape),
                                                batch_size)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dws = dict()
    ew_vars_inp, ew_vars_hid = elg_trace_vars
    _, weights, elt_vars = variables
    for wkey in weights:
      w = weights[wkey]  # weight
      dx = ew_vars_inp[wkey]  # input eligibility trace
      op = self.all_vars[wkey].op  # operator of the weight

      for ekey in elt_vars.keys():
        key = self._elt_key(ekey, wkey)  # eligibility trace key
        if key in ew_vars_hid:
          # the hidden eligibility trace
          dy = jax.vmap(
            lambda hid: jax.vmap(jnp.matmul)(dg_elt_vars[ekey], hid),  # BATCH axis vmapping
            in_axes=1, out_axes=1
          )(ew_vars_hid[key])  # RANK axis vmapping
          _update_dict(dws, wkey, _vmap_op_auto_grad(op, dx, w, dy, map_axis=1))
    dws = {k: (v) for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, *args):
    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    def _get_hidden_etrace_of_given_weight(w_key):
      res = dict()
      for e_key, elt_data in et_vars_data.items():
        ew_key = self._elt_key(e_key, w_key)
        hid_trace = (
          self.elg_trace_hid[ew_key].value  # [B, R, N, N]
          if ew_key in self.elg_trace_hid else
          jnp.zeros((elt_data.shape[0], self.num_rank, elt_data.shape[1], elt_data.shape[1]))  # [B, R, N, N]
        )
        res[e_key] = hid_trace
      return res

    # the function returns the given eligibility trace variable
    f_for_jac = lambda et_vars: self._update_all_elt_vars(weights_data, et_vars, oth_vars_data, args)

    for w_key in weights_data:
      # the weight gradients of the given eligibility trace variable
      eweights = _get_hidden_etrace_of_given_weight(w_key)

      # [JVP] jacobian vector product #
      # Therefore, using the "jax.jvp" function is enough.
      jvp_res = jax.vmap(
        lambda ewss: jax.vmap(lambda ews: jax.jvp(f_for_jac, (et_vars_data,), (ews,))[1], in_axes=2, out_axes=2)(ewss),
        in_axes=1, out_axes=1
      )(eweights)

      # the hidden to hidden Jacobian transition
      for elt_key in et_vars_data:
        key = self._elt_key(elt_key, w_key)
        if key in self.elg_trace_hid:
          self.elg_trace_hid[key] = jvp_res[elt_key]

      # recovery data
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # perturb data
    bs = args[0].shape[0]
    perturb_data = {k: jnp.zeros((bs,) + v.yshape) for k, v in self.all_vars.subset(ETraceParam).items()}
    # vector-valued gradients computing the right cotangents
    yy, f_vjp, inputs = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # [KEY]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        self.elg_trace_inp[w_key] = _shift_with_batch(self.elg_trace_inp[w_key], inputs[w_key])

    # [KEY]
    # update the right cotangents
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v))
                  for k, v in yy.items()}
      dfs = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          self.elg_trace_hid[ew_key] = _shift_with_batch(self.elg_trace_hid[ew_key], dfs[w_key])

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())


class TruncatedDiagRTRL(_RTRLWithDiag):
  """Real-Time Recurrent Learning with Truncated Gradient Length.

  This method is also called Truncated Real-Time Recurrent Learning (TRTRL). It uses the truncated
  gradient length to approximate the hidden to hidden Jacobian transition.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).
  """

  def __init__(
      self,
      layer: bc.Module,
      num_rank: int,
      jac_type: Union[str, JacobianApprox] = JacobianApprox.diagonal,
      **kwargs
  ):
    super().__init__(layer, **kwargs)

    # Jacobian type
    self.jac_type = JacobianApprox.get(jac_type)

    # number of approximation rank
    self.num_rank = num_rank

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the Jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            ew_key = self._elt_key(et_key, w_key)
            self.elg_trace_hid[ew_key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.yshape), batch_size)
            count += 1

        # input variable
        if count > 0:
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.xshape), batch_size)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dws = dict()
    elg_keys = tuple(self.all_vars.subset(ETraceVar).keys())
    ew_vars_inp, ew_vars_hid = elg_trace_vars
    _, weights, _ = variables
    for wkey in weights:
      w = weights[wkey]  # weight
      dx = ew_vars_inp[wkey]  # input eligibility trace
      op = self.all_vars[wkey].op  # operator

      for ekey in elg_keys:
        key = self._elt_key(ekey, wkey)
        if key in ew_vars_hid:
          dy = ew_vars_hid[key] * jnp.expand_dims(dg_elt_vars[ekey], 1)
          _update_dict(dws, wkey, _vmap_op_auto_grad(op, dx, w, dy))
    dws = {k: (v) for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, *args):
    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    # the function returns the given eligibility trace variable
    f_for_jac = lambda et_vars: self._update_all_elt_vars(weights_data, et_vars, oth_vars_data, args)

    if self.jac_type == JacobianApprox.diagonal:
      # approximate the hidden to hidden Jacobian diagonal using the JVP
      self._stop_weight_gradients(True)
      diagonal = bm.vector_grad(f_for_jac, argnums=0)(et_vars_data)
      # ones = jax.tree_map(jnp.ones_like, et_vars_data)
      # diagonal = jax.jvp(f_for_jac, (et_vars_data,), (ones,))[1]
      self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)  # recovery data
      self._stop_weight_gradients(False)

      # update the hidden to hidden Jacobian
      for w_key in weights_data:
        for elt_key in et_vars_data:
          key = self._elt_key(elt_key, w_key)
          if key in self.elg_trace_hid:
            self.elg_trace_hid[key].value = jnp.expand_dims(diagonal[key], 1) * self.elg_trace_hid[key]
    else:
      def _get_hidden_etrace_of_given_weight(wkey):
        res = dict()
        for e_key, elt_data in et_vars_data.items():
          ew_key = self._elt_key(e_key, wkey)
          hid_trace = (self.elg_trace_hid[ew_key].value
                       if ew_key in self.elg_trace_hid else
                       jnp.zeros((elt_data.shape[0], self.num_rank) + elt_data.shape[1:]))
          res[e_key] = hid_trace
        return res

      for w_key in weights_data:
        # the weight gradients of the given eligibility trace variable
        eweights = _get_hidden_etrace_of_given_weight(w_key)

        # jacobian vector product #
        # ----------------------- #
        # [NOTE]
        # For the rank-one approximation,
        # the "etrace_to_weights" has the same shape of the "et_vars_data".
        # Therefore, using the "jax.jvp" function is enough.
        jvp_res = jax.vmap(lambda ews: jax.jvp(f_for_jac, (et_vars_data,), (ews,))[1],
                           in_axes=1, out_axes=1)(eweights)
        self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)  # recovery data

        # the hidden to hidden Jacobian transition
        for elt_key in et_vars_data:
          key = self._elt_key(elt_key, w_key)
          if key in self.elg_trace_hid:
            self.elg_trace_hid[key] = jvp_res[elt_key]

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # perturb data
    bs = args[0].shape[0]
    perturb_data = {k: jnp.zeros((bs,) + v.yshape) for k, v in self.all_vars.subset(ETraceParam).items()}
    # vector-valued gradients computing the right cotangents
    yy, f_vjp, inputs = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # [KEY]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        self.elg_trace_inp[w_key] = _shift_with_batch(self.elg_trace_inp[w_key], inputs[w_key])

    # [KEY]
    # update the right cotangents
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      dfs = f_vjp(tangents)[0]

      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          self.elg_trace_hid[ew_key] = _shift_with_batch(self.elg_trace_hid[ew_key], dfs[w_key])

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())


class TruncatedLoraDiagRTRL(_RankNRTRL):
  """Real-Time Recurrent Learning with Truncated Gradient Length.

  This method is also called Truncated Real-Time Recurrent Learning (TRTRL). It uses the truncated
  gradient length to approximate the hidden to hidden Jacobian transition.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).
  """

  def __init__(self, layer: bc.Module, num_rank: int, **kwargs):
    super().__init__(layer, **kwargs)

    # number of approximation rank
    self.num_rank = num_rank

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():
        # batch size

        count = 0
        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the Jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            key = self._elt_key(et_key, w_key)
            # current hidden jacobian f'
            self.elg_trace_hid[key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.yshape), batch_size)
            # previous hidden jacobian f'
            self.elg_trace_hid_prev[key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.yshape), batch_size)
            # LORA hidden lefts
            self.elg_trace_lora_lefts[key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.yshape), batch_size)
            # LORA hidden rights
            self.elg_trace_lora_rights[key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.yshape), batch_size)
            count += 1

        # input variable
        if count > 0:
          # current input
          self.elg_trace_inp[w_key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.xshape), batch_size)
          # previous input
          self.elg_trace_inp_prev[w_key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.xshape), batch_size)
          # LORA inputs
          self.elg_trace_lora_inp[w_key] = init.state(jnp.zeros, (self.num_rank,) + tuple(w_var.xshape), batch_size)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    raise NotImplementedError
    dws = dict()
    elg_keys = tuple(self.all_vars.subset(ETraceVar).keys())
    elg_trace_inp, elg_trace_hid, elg_trace_lora_inp, elg_trace_lora_lefts, elg_trace_lora_rights = elg_trace_vars
    _, weights, _ = variables
    for w_key in weights:
      w = weights[w_key]  # weight
      op = self.all_vars[w_key].op  # operator of the weight
      dx = elg_trace_inp[w_key]  # input eligibility trace of LORA
      dx_lora = elg_trace_lora_inp[w_key]  # input eligibility trace of LORA

      for ekey in elg_keys:
        key = self._elt_key(ekey, w_key)
        if key in elg_trace_hid:
          # the gradients computed by the low-ranked approximation
          dy_lora = (
              _inner((dg_elt_vars[ekey]),  # loss
                     (elg_trace_lora_lefts[key]),  # LORA lefts
                     keepdims=True)
              * (elg_trace_lora_rights[key])  # LORA rights
          )
          _update_dict(dws, w_key, _op_auto_grad(op, dx_lora, w, dy_lora))

          # the gradients computed by the diagonal approximation
          dy = (dg_elt_vars[ekey]) * elg_trace_hid[key]
          _update_dict(dws, w_key, _op_auto_grad(op, dx, w, dy))

    dws = {k: (v) * self.num_rank for k, v in dws.items()}
    return dws

  def _update_eligibility_trace(self, args):
    """Separate the hidden to weight Jacobian into two parts: left and right cotangents.

    This function is the key, since the separation of left and right
    cotangents is the key for the low-rank approximation.

    Args:
      args: The arguments for the layer.
    """
    raise NotImplementedError

    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      lefts = self._collect_x_primals(weights_data.keys())
      return new_et_vars, lefts

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)
    # vector-valued gradients computing the right cotangents
    yy, f_vjp, left_cots = jax.vjp(f_perturb, perturb_data, has_aux=True)
    self._record_x_primal(False)

    # [KEY]
    # update the left cotangents
    for w_key in weights_data:
      if w_key in self.elg_trace_inp:
        a = self.elg_trace_inp[w_key][:, 1:]
        b = bm.expand_dims(left_cots[w_key], axis=1)
        self.elg_trace_inp[w_key] = (bm.concat([a, b], axis=1))

    # [KEY]
    # update the right cotangents
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      right_cotangents = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          a = self.elg_trace_hid[ew_key][:, 1:]
          b = bm.expand_dims(right_cotangents[w_key], axis=1)
          self.elg_trace_hid[ew_key] = (bm.concat([a, b], axis=1))

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())


class OptLoraRTRL(_RankNRTRL):
  """
  Real-Time Recurrent Learning (RTRL) with the optimal rank-one approximation of the gradient matrix.

  This training algorithm uses the QR factorization and singular value decomposition (SVD)
  to approximate the gradient matrix. The approximation is optimal in the sense of the Frobenius
  norm. The approximation is also optimal in the sense of the spectral norm if the gradient
  matrix is a square matrix.

  Notably, this method should be used in reverse-mode automatic differentiation (AD).
  """

  def __init__(self, layer: bc.Module, num_rank: int, **kwargs):
    super().__init__(layer, num_rank=num_rank, **kwargs)

  @bc.call_order(-1)
  def reset_state(self, batch_size):
    elt_vars, train_vars = super().reset_state()

    if self.mode.is_train_mode():
      # eligibility trace: one variable for one weight
      for w_key, w_var in train_vars.items():

        # hidden variable
        for et_key, et_var in elt_vars.items():
          # remove batch size
          et_shape = et_var.shape[1:]

          # Ignore the trace of the hidden variable to the weight.
          # If the shape of the hidden variable is not equal to
          # the shape of weight's hidden eligibility trace, then
          # the jacobian matrix of the hidden to the weight should
          # be the zero matrix.
          if et_shape == w_var.yshape:
            ew_key = self._elt_key(et_key, w_key)
            self.elg_trace_hid[ew_key] = init.state(jnp.zeros, (self.num_rank,) + w_var.yshape, batch_size)
            self.elg_trace_inp[ew_key] = init.state(jnp.zeros, (self.num_rank,) + w_var.xshape, batch_size)

  def _weight_backward(self, dg_elt_vars, elg_trace_vars, variables, args) -> dict:
    dg_elt_vars = jax.tree_util.tree_map(functools.partial(jnp.expand_dims, axis=1), dg_elt_vars)
    dws = dict()
    elg_keys = tuple(self.all_vars.subset(ETraceVar).keys())
    ew_vars_inp, ew_vars_hid = elg_trace_vars
    _, weights, _ = variables
    for wkey in weights:
      ekey = elg_keys[0]  # eligibility trace key
      w = weights[wkey]  # weight
      op = self.all_vars[wkey].op  # operator
      key = self._elt_key(ekey, wkey)
      if key in ew_vars_hid:
        dx = ew_vars_inp[key]  # input eligibility trace
        dy = ew_vars_hid[key] * dg_elt_vars[ekey]  # hidden eligibility trace
        dws[wkey] = _vmap_op_auto_grad(op, dx, w, dy)  # gradients of the weight
      for ekey in elg_keys[1:]:
        key = self._elt_key(ekey, wkey)
        if key in ew_vars_hid:
          dx = ew_vars_inp[key]  # input eligibility trace
          dy = ew_vars_hid[key] * dg_elt_vars[ekey]  # hidden eligibility trace
          _update_dict(dws, wkey, _vmap_op_auto_grad(op, dx, w, dy))
    dws = {k: (v) for k, v in dws.items()}
    return dws

  def _get_jac_of_hidden_to_weight(self, perturb_data, args):
    """Separate the hidden-to-weight Jacobian into two parts: left and right cotangents.

    This function is the key, since the separation of left and right
    cotangents is the key for the low-rank approximation.

    Args:
      perturb_data: The perturbation data for the layer hidden variables.
      args: The arguments for the layer.
    """

    weights_data, et_vars_data, oth_vars_data = self._get_variable_data()

    i = bc.share.load('i', desc='The index of the running iteration. ')

    def f_perturb(perturbs):
      # assign perturb data
      self._assign_perturb_data(perturbs)
      # call the model
      new_et_vars = self._update_all_elt_vars(weights_data, et_vars_data, oth_vars_data, args)
      # get left primal values
      left_ps = self._collect_x_primals(weights_data.keys())
      return new_et_vars, left_ps

    # record the left primals during the first update of the target layer
    self._record_x_primal(True)

    # vector-valued gradients computing the right cotangents
    yy, f_vjp, left_cots = jax.vjp(f_perturb, perturb_data, has_aux=True)
    # turn off the recording of the left primals
    self._record_x_primal(False)

    # [KEY]
    # update the left and right cotangents
    for elt_key in et_vars_data:
      tangents = {k: (jnp.ones_like(v) if k == elt_key else jnp.zeros_like(v)) for k, v in yy.items()}
      right_cotangents = f_vjp(tangents)[0]
      for w_key in weights_data:
        ew_key = self._elt_key(elt_key, w_key)
        if ew_key in self.elg_trace_hid:
          # JVP cannot differentiate the "jnp.linalg.qr" and "jnp.linalg.svd" function.
          lefts, rights = _update_elg_trace(
            i,
            jax.lax.stop_gradient(self.elg_trace_inp[ew_key]),
            jax.lax.stop_gradient(left_cots[w_key]),
            jax.lax.stop_gradient(self.elg_trace_hid[ew_key]),
            jax.lax.stop_gradient(right_cotangents[w_key])
          )
          self.elg_trace_inp[ew_key] = (lefts)
          self.elg_trace_hid[ew_key] = (rights)

    # recovery data
    self._assign_variable_data(weights_data, et_vars_data, oth_vars_data)
    self._clear_perturb_data(weights_data.keys())
