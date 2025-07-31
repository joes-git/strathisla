# Statisical Model Plugins for Spey

Various plugins for [Spey](https://github.com/SpeysideHEP/spey)

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

## `SimpleMultivariateGaussianEFT`

Simple multivariate Gaussian likelihood (no nuisance parameters) for a histogram with $i$ correlated bins and two signal contributions: A term $s_{\text{lin}}$ with scales linearly with the signal strength $\mu$, and a term $s_{\text{quad}}$, which scales quadratically $\mu^2$ The form of this likelihood is:

$$
L(\mu) = 
\frac{1}{\sqrt{(2\pi)^k \det(\Sigma)}}
\exp \left( -\frac{1}{2} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n)^{\text{T}} \Sigma^{-1} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n) \right),
$$

where $\Sigma$ is the covariance matrix, $b$ is the background and $n$ is the data.

## `MultivariateGaussianCovarianceScaledEFT`

The multivariante Gaussian likelihood above, but the signal contributions to the covariance matrix are scaled by the parameter of interest as:

$$
\Sigma(\mu) = \mu^4 \Sigma_{\text{quad}} + \mu^2 \Sigma_{\text{lin}} + \Sigma_b + \Sigma_n
$$
