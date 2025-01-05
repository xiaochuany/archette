# archette

GARCH is a minimal implementation of the GARCH(1,1) model with zero
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

    array([1.46162489e+00, 0.00000000e+00, 4.34165554e-16])

The conditional variance of the fit model is computed by the recursion

$$
\hat{\sigma}_t^2 = \hat{\omega} + \hat{\alpha} + \hat{\beta} \hat{\sigma}_{t-1}^2
$$

which depends entirely on the unobserved $\sigma_0^2$. The model sets a
sensible default value for it.

``` python
mod.vs
```

    array([1.6637509 , 1.46162489, 1.46162489, 1.46162489, 1.46162489,
           1.46162489, 1.46162489, 1.46162489, 1.46162489, 1.46162489,
           1.46162489, 1.46162489, 1.46162489, 1.46162489, 1.46162489,
           1.46162489, 1.46162489, 1.46162489, 1.46162489, 1.46162489])

The standardised residual is deduced from the conditional variance

$$
r_t = \frac{y_t}{\hat{\sigma}_t}
$$

``` python
mod.std_resids
```

    array([ 1.83606806,  0.40202783, -0.05913285, -0.47174874, -0.59056739,
           -2.20575562,  0.71528908,  0.26541164, -0.20129438,  1.33214741,
           -0.50849529,  0.2951227 , -0.84969512,  1.53542218,  0.44267321,
           -1.95023795, -1.68947028, -0.23377285, -0.18693467,  0.7959041 ])

Finally, one can forecast the condtional variance and simulate the
process with the fit parameters with a given horizon

``` python
mod.forecast_vs(horizon=5)
```

    array([1.46162489, 1.46162489, 1.46162489, 1.46162489, 1.46162489])

``` python
mod.simulate(horizon=5, method="bootstrap", n_rep=2, seed=1)
```

    array([[-2.66670736,  0.35679649, -1.02726169, -0.24336023,  1.61053531],
           [ 0.35679649, -2.66670736, -2.35779243,  2.21976368, -2.04253036]])
           