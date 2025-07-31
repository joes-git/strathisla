import pytest
import numpy as np

from spey.system.exceptions import InvalidInput

from strathisla.nuisance_parameters import FullNuisanceParameters

# define some good data that should pass the constructor
three_bin_yields = np.array([1.0,2.0,3.0])
three_bin_cov = np.identity(3)

def test_empty_input_raise():
    empty_input = []
    with pytest.raises(InvalidInput) as excinfo:
        FullNuisanceParameters(empty_input,three_bin_yields,three_bin_yields,three_bin_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Inputs must not be empty'

def test_not_list_raise():
    with pytest.raises(InvalidInput) as excinfo:
        FullNuisanceParameters(5.0,three_bin_yields,three_bin_yields,three_bin_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Pass input arguments as lists or numpy arrays'

def test_different_yield_lengths_raise():
    one_bin_yield = np.array([9.0])
    with pytest.raises(InvalidInput) as excinfo:
        FullNuisanceParameters(one_bin_yield,three_bin_yields,three_bin_yields,three_bin_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Yields must be the same length'

def test_wrong_size_cov_raise():
    two_bin_cov = np.identity(2)
    with pytest.raises(InvalidInput) as excinfo:
        FullNuisanceParameters(three_bin_yields,three_bin_yields,three_bin_yields,two_bin_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Covariance matrices size should match the number of yields'

def test_covariance_not_two_dimensional_raise():
    higher_dim_cov = np.ones((len(three_bin_yields),len(three_bin_yields), 5)) # a 3x3x5 matrix of ones
    with pytest.raises(InvalidInput) as excinfo:
        FullNuisanceParameters(three_bin_yields,three_bin_yields,three_bin_yields,higher_dim_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == '2D covariance matrix required'

def test_covariance_not_square_raise():
    non_square_cov = np.ones((len(three_bin_yields),len(three_bin_yields)+2)) # a 3x5 matrix of ones
    with pytest.raises(InvalidInput) as excinfo:
        FullNuisanceParameters(three_bin_yields,three_bin_yields,three_bin_yields,non_square_cov,three_bin_cov,three_bin_cov)
    assert str(excinfo.value) == 'Covariance matrix must be square'