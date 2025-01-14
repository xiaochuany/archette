import jax 
import jax.numpy as jnp
from jax import jit, grad, vmap, random

key = random.PRNGKey(0)
ws = random.normal(key, (3,3))

print(ws)