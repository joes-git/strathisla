# Spey Likelihood Prescriptions for Contur

Spey plugins developed for likelihood calculations in the [contur](https://gitlab.com/hepcedar/contur) toolkit.

Plugins:

- [`ConturHistogram`](#conturhistogram): For full profiling a histogram including all nuisance parameters
- [`MultivariateGaussianEFT`](#multivariategaussianeft): Gaussian likelihood that is quadratic in the signal strength parameter 

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
git clone git@github.com:joes-git/contur_likelihood.git
```

2. Add this repo to your environment
```
cd contur_likelihood
pip install -e .
```

3. Check the plugins are available in Spey
```python
import spey
contur_plugins = [
    'contur.full_histogram_likelihood',
    'contur.multivariate_gaussian_eft'
]

print(all(plugin in spey.AvailableBackends() for plugin in contur_plugins))
# True
```
 
## `ConturHistogram` 

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

## `MultivariateGaussianEFT`

Simple Gaussian likelihood (no nuisance parameters) for a histogram with $i$ correlated bins and two signal contributions: An interference term $s_{\text{int}}$ with scales linearly with the signal strength $\mu$, and a square (or pure BSM) term $s_{\text{sq}}$, which scales quadratically ($\mu^2$) The form of this likelihood is:
$$
L(\mu) = 
\frac{1}{\sqrt{(2\pi)^k \det(\Sigma)}}
\exp \left( -\frac{1}{2} (\mu^2 s_{\text{sq}} + \mu s_{\text{int}} + b - n)^{\text{T}} \Sigma^{-1} (\mu^2 s_{\text{sq}} + \mu s_{\text{int}} + b - n) \right),
$$
where $\Sigma$ is the covariance matrix, $b$ is the background and $n$ is the data.