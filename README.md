# Spey Likelihood Prescriptions for Contur

Spey plugin for the likelihood presented in [arXiv:2102.04377](https://arxiv.org/pdf/2102.04377.pdf)
 
Developed for likelihood calculations in the [contur](https://gitlab.com/hepcedar/contur) toolkit

`ConturHistogram` represents the likelihood for a histogram with $i$ bins and covariance matrices ($\Sigma$) for the signal ($s$), background ($b$) and data ($n$) yields. This likelihood has the following form:

$$
L(\mu, \theta) = \prod_{i \in {\rm bins}} 
        {\rm Poiss} ( n_i \vert \mu s_i+b_i + \sum_{j \in n,s,b} \theta^{(j)}_i \sigma^{(j)}_i)
        \prod_{j \in n,s,b} 
        {\rm Gauss}(\theta^{(j)}|0,\Sigma^{(j)})
$$

where $\mu$ is the parameter of interest, and $\theta$ are nuisance parameters.

## Installation

To use this plugin with Spey:

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

3. Check the plugin is available in Spey
```python
import spey
print('contur.full_histogram_likelihood' in spey.AvailableBackends())

# True
```