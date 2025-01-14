import jax 
import jax.numpy as jnp
from jax import jit, grad, vmap, random, lax

from functools import partial

@partial(jit, static_argnames=("horizon",))
def simulate(params, last_y, horizon, key):
    om, al = params
    def one_step(carry, ep):
        y = carry
        new_y = om + al * y + ep
        return new_y, new_y
    ks = random.split(key, horizon)
    ws = vmap(random.normal)(ks)
    _, ys = lax.scan(one_step, last_y, ws)
    return ys

key = random.key(42)
params = jnp.array([1.0, 0.1])
last_y = 1.0
ys = simulate(params, last_y, 4, key)

print(ys)