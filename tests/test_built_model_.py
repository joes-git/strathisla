import pytest
import numpy as np

from spey.backends.distributions import Normal, MultivariateNormal

from contur_likelihood.likelihoods import ConturHistogram

def test_general_model_types():
    two_bins = np.ones(2)
    two_bins_cov = np.identity(2)
    ch = ConturHistogram(two_bins,two_bins,two_bins,
                         two_bins_cov,two_bins_cov,two_bins_cov)
    assert ch.main_model.pdf_type=='poiss'
    assert all([isinstance(model,MultivariateNormal) for model in ch.constraint_model._pdfs])

def test_one_bin_model_types():
    one_bin = np.array([1.0])
    ch = ConturHistogram(one_bin,one_bin,one_bin,one_bin,one_bin,one_bin)
    assert ch.main_model.pdf_type=='poiss'
    assert all([isinstance(model,Normal) for model in ch.constraint_model._pdfs])