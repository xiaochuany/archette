# archette

Let's build arch-like models from scractch. 

This includes model specification, maximum likelihood estimation, etc. 

The current implementation is with `numpy` for array manipulations, `scipy` for optimization of objectives, 
and `numba` for acceleration. 

TODO: swap `numpy`, `scipy` with `jax` and optimization with by hand in place of the high level API of `scipy.optimize`.   

## API Reference

::: archette.garch.GARCH