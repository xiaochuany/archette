import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from functools import partial
from typing import Optional

@jit
def _get_vs(y, params, init=1.0):
    om, al, be = params

    def one_step(carry, x):
        new_carry = om + al * x**2 + be * carry
        return new_carry, new_carry

    _, vs = jax.lax.scan(one_step, init=init, xs=y)
    return vs


@jit
def _nllgauss(y: jnp.ndarray, params: jnp.ndarray, sig2_init: float) -> float:
    vs = _get_vs(y, params, sig2_init)
    nll = jnp.log(vs) + y**2 / vs
    return nll.sum()


@partial(jit, static_argnames=("horizon",))
def _make_fcst_vs(
    params: np.ndarray, last_v: float, last_y: float, horizon: int
) -> jnp.ndarray:
    om, al, be = params
    init = om + al * last_y**2 + be * last_v
    def one_step(carry, _):
        new_carry = om + (al + be) * carry
        return new_carry, new_carry

    _, fcst_vs = jax.lax.scan(
        one_step, init=init, xs=None, length=horizon
    )
    return fcst_vs

@partial(jit, static_argnames=("horizon","n_rep","seed"))
def _simulate(
    params: jnp.ndarray,
    last_y: float,
    last_v: float,
    horizon: int,
    n_rep: int,
    seed: int,
    ws: Optional[np.ndarray] = None,
) -> np.ndarray:
    om, al, be = params
    if ws is None:
        np.random.seed(seed)
        ws = np.random.randn(n_rep, horizon)
    
    def one_step(carry, x):
        y,v = carry
        new_v = om + al * y**2 + be * v
        new_y = jnp.sqrt(new_v) * x
        return (new_y, new_v), new_y
    
    _, ys = jax.lax.scan(one_step, init=(last_y, last_v), xs=ws)
    return ys

om, al, be = 1.0, 0.1, 0.8

last_y = 1.0
last_v = 1.0
horizon= 4
n_rep = 2
ws = np.random.randn(horizon)


def one_step(carry, x):
    y,v = carry
    new_v = om + al * y**2 + be * v
    new_y = jnp.sqrt(new_v) * x
    return (new_y, new_v), new_y
    
_, ys = jax.lax.scan(one_step, init=(last_y, last_v), xs=ws)

print(ys)

# print(_get_vs(jnp.array([1, 2, 3]), jnp.array([om, al, be])))
# print(_nllgauss(jnp.array([1, 2, 3]), jnp.array([om, al, be]), 1.0))
# print(_make_fcst_vs(jnp.array([om, al, be]), last_v, last_y, horizon))
# print(_simulate(jnp.array([om, al, be]), last_y, last_v, horizon, n_rep=2, seed=0, ws=ws))