# Statisical Model Plugins for Spey

[![Tests](https://github.com/joes-git/strathisla/actions/workflows/tests.yml/badge.svg)](https://github.com/joes-git/strathisla/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/joes-git/strathisla.svg)](https://github.com/joes-git/strathisla/blob/main/LICENSE)

Various likelihood plugins for [Spey](https://github.com/SpeysideHEP/spey). The package is named after the [Strathisla](https://en.wikipedia.org/wiki/Strathisla_distillery), the Speyside distillery.

Available models:

- [`FullNuisanceParameters`](#fullnuisanceparameters): Poisson likelihood with nuisance parameters on the signal, background and data.
- [`SimpleMultivariateGaussianEFT`](#simplemultivariategaussianeft): Multivariate Gaussian likelihood with no nuisance parameters. Two signal inputs: the first scales linearly with the parameter of interest, the second quadratically.
- [`MultivariateGaussianCovarianceScaledEFT`](#multivariategaussiancovariancescaledeft): As above, but the signal covariance matrices are scaled by the parameter of interest

## Installation

To use these plugins with Spey:

0. (Optional) Setup a virtual environment
```
python -m venv spey_venv
source spey_venv/bin/activate
pip install spey
```

1. Clone this repo
```
git@github.com:joes-git/strathisla.git
```

2. Add this repo to your environment
```
cd strathisla
pip install -e .
```

3. Check the plugins are available in Spey
```python
import spey
installed_plugins = [
    'strathisla.full_nuisance_parameters',
    'strathisla.simple_multivariate_gaussian_eft',
    'strathisla.multivariate_gaussian_scaled_covariance_eft'

]

print(all(plugin in spey.AvailableBackends() for plugin in installed_plugins))
# True
```
 
## `FullNuisanceParameters` 

From equation 8 of [arXiv:2102.04377](https://arxiv.org/pdf/2102.04377.pdf), this is the likelihood for a histogram with $i$ bins and covariance matrices ($\Sigma$) for the signal ($s$), background ($b$) and data ($n$) yields. This likelihood has the following form:

$$
L(\mu, \theta) = 
\prod_{i \in {\rm bins}} 
{\rm Poiss}
(n_i \vert \mu s_i+b_i + \sum_{j \in n,s,b}  \theta_i^{(j)} \sigma_i^{(j)})
\prod_{j \in n,s,b} 
{\rm Gauss}(\theta^{(j)}|0,\Sigma^{(j)})
$$

where $\mu$ is the parameter of interest, and $\theta$ are nuisance parameters.

### Example
```python
import spey
import numpy as np

data = np.array([30, 35, 40])
signal_yields = np.array([8.0, 10.0, 12.0])
background_yields = np.array([25.0, 28.0, 32.0])
    
signal_covariance = np.array([[1.0, 0.2, 0.1], 
                              [0.2, 1.5, 0.3], 
                              [0.1, 0.3, 2.0]])
    
background_covariance = np.array([[16.0, 4.0, 2.0],
                                  [4.0, 25.0, 5.0],
                                  [2.0, 5.0, 36.0]])
    
data_covariance = np.array([[30.0,2.0,0.5],
                            [4.3, 35.0, 9.0],
                            [6.2, 13.0, 40.0]])

backend = spey.get_backend("strathisla.full_nuisance_parameters")

stat_model = backend(
    signal_yields=signal_yields,
    background_yields=background_yields,
    data=data,
    signal_covariance=signal_covariance,
    background_covariance=background_covariance,
    data_covariance=data_covariance
)

CLs = stat_model.exclusion_confidence_level()[0]
print(CLs)
```

## `SimpleMultivariateGaussianEFT`

Simple multivariate Gaussian likelihood (no nuisance parameters) for a histogram with $i$ correlated bins and two signal contributions: A term $s_{\text{lin}}$ with scales linearly with the signal strength $\mu$, and a term $s_{\text{quad}}$, which scales with $\mu^2$. The form of this likelihood is:

$$
L(\mu) = 
\frac{1}{\sqrt{(2\pi)^k \det(\Sigma)}}
\exp \left( -\frac{1}{2} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n)^{\text{T}} \Sigma^{-1} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n) \right),
$$

where $\Sigma$ is the covariance matrix, $b$ is the background and $n$ is the data.

### Example
```python
import spey
import numpy as np

quadratic_term = np.array([9.908565e-04, 1.082060e-02, 4.402903e-02])
linear_term = np.array([0.00045349, 0.00304028, 0.0090483])
background = np.array([0.25291, 0.59942, 1.52996])
data = np.array([0.2325,  0.5385,  1.25734])

covariance = np.array([
        [0.06427483, 0.05824, 0.01225],
        [0.05824, 0.06694904, 0.01207],
        [0.01225, 0.01207, 0.02126036]])

backend = spey.get_backend("strathisla.simple_multivariate_gaussian_eft")

stat_model = backend(
    quadratic_term=quadratic_term,
    linear_term=linear_term,
    background=background,
    data=data,
    covariance=covariance
)
    
CLs = stat_model.exclusion_confidence_level()

print(CLs[0])
```

## `MultivariateGaussianCovarianceScaledEFT`

The multivariante Gaussian likelihood above, but the signal contributions to the covariance matrix are scaled by the parameter of interest as:

$$
\Sigma(\mu) = \mu^4 \Sigma_{\text{quad}} + \mu^2 \Sigma_{\text{lin}} + \Sigma_b + \Sigma_n
$$

### Example
```python
import spey
import numpy as np

quadratic_term = np.array([9.908565e-04, 1.082060e-02, 4.402903e-02])
linear_term = np.array([0.00045349, 0.00304028, 0.0090483])
background = np.array([0.25291, 0.59942, 1.52996])
data = np.array([0.2325,  0.5385,  1.25734])

covariance = np.array([
        [0.06427483, 0.05824, 0.01225],
        [0.05824, 0.06694904, 0.01207],
        [0.01225, 0.01207, 0.02126036]])

background_covariance = np.zeros_like(covariance) # assume 'covariance' has both data and background

quadratic_term_covariance = np.diag([1.42421005e-07, 8.57147040e-07, 1.99661509e-06])
linear_term_covariance = np.diag([3.60939969e-08, 4.65573038e-07, 9.61593099e-07])

backend = spey.get_backend("strathisla.multivariate_gaussian_scaled_covariance_eft")

stat_model = backend(
    quadratic_term=quadratic_term,
    linear_term=linear_term,
    background=background,
    data=data,
    data_covariance=covariance,
    background_covariance=background_covariance,
    quadratic_term_covariance=quadratic_term_covariance,
    linear_term_covariance=linear_term_covariance
)
    
CLs = stat_model.exclusion_confidence_level()

print(CLs[0])
```