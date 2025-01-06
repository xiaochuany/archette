# archette

Let's build arch-like models from scratch. 

This includes model specification, maximum likelihood estimation, etc. 

The current implementation is to use `numpy` for array manipulations, `scipy` for optimization of objectives, 
and `numba` for acceleration. 

TODO: reduce dependency to  `jax` alone, with optimization done by hand (gradient descent and the likes) in place of the high level API of `scipy.optimize`.   

## API Reference

::: archette.garch.GARCH