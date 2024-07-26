import pytest
import numpy as np

import spey
from spey.system.exceptions import InvalidInput

from contur_likelihood.likelihoods import ConturHistogram

# define some good data that should pass the constructor
three_bin_yields = np.array([1.0,2.0,3.0])
three_bin_cov = np.identity(3)

def test_empty_input_raise():
    empty_input = []
    with pytest.raises(InvalidInput) as excinfo:
        ConturHistogram(empty_input,three_bin_yields,three_bin_yields,three_bin_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Inputs must not be empty'

def test_not_list_raise():
    with pytest.raises(InvalidInput) as excinfo:
        ConturHistogram(5.0,three_bin_yields,three_bin_yields,three_bin_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Pass input arguments as lists or numpy arrays'

