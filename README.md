# archette


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

GARCHETTE is a minimal implementation of the GARCH(1,1) model with zero
mean and Gaussian noise.

## Usage

### Installation

Install latest from the GitHub
[repository](https://github.com/xiaochuany/archette):

``` sh
$ pip install git+https://github.com/xiaochuany/archette.git
```

### Documentation

Documentation can be found hosted on this GitHub
[repository](https://github.com/xiaochuany/archette)’s
[pages](https://xiaochuany.github.io/archette/).

## How to use

``` python
import numpy as np
from archette import GARCHETTE
```

``` python
y = np.random.randn(20)
mod = GARCHETTE()
mod.fit(y)
```

    <archette.core.GARCHETTE>

One can get inspect the fit params: a triplet of omega, alpha and beta
as in

$$
\begin{align*}
Y_t &= \sigma_t W_t \\
\sigma_t^2 &=  \omega + \alpha Y_{t-1}^2 + \beta \sigma^2_{t-1} 
\end{align*}
$$

``` python
mod.params
```

    array([1.04589261e-08, 0.00000000e+00, 9.59250037e-01])

The conditional variance of the fit model is computed by the recursion

$$
\hat{\sigma}_t^2 = \hat{\omega} + \hat{\alpha} + \hat{\beta} \hat{\sigma}_{t-1}^2
$$

which depends entirely on the unobserved $\sigma_0^2$. The model sets a
sensible default value for it.

``` python
mod.vs
```

    array([1.16741485, 1.11984275, 1.07420921, 1.03043524, 0.98844505,
           0.94816596, 0.90952825, 0.87246501, 0.83691211, 0.80280798,
           0.7700936 , 0.73871232, 0.70860983, 0.67973402, 0.65203489,
           0.62546451, 0.59997686, 0.57552784, 0.55207511, 0.52957808])

The standardised residual is deduced from the conditional variance

$$
r_t = \frac{y_t}{\hat{\sigma}_t}
$$

``` python
mod.std_resids
```

    array([-1.48243023, -1.39651573,  0.46649642,  0.18563289, -0.46739453,
            0.88507072, -2.78314409,  1.23361127,  0.214951  , -1.72457798,
           -0.10039782,  0.99157118,  0.8045169 , -0.12963449,  0.89818693,
            1.18211244,  1.44374303,  0.07330686,  0.5069283 ,  0.51686696])

Finally, one can forecast the condtional variance and simulate the
process with the fit parameters with a given horizon

``` python
mod.forecast_vs(horizon=5)
```

    array([0.5079978 , 0.48729692, 0.4674396 , 0.44839146, 0.43011954])

``` python
mod.simulate(horizon=5, method="bootstrap", n_rep=2, seed=1)
```

    array([[ 0.63082499,  0.69218269,  0.55004462,  0.14393557, -1.13103859],
           [ 0.70673209,  0.61783827,  0.80820501, -0.99266551,  0.9468572 ]])
