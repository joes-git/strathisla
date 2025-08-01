"""Test nuisance parameters functionality"""

import numpy as np
import pytest
import spey
from spey.backends.distributions import Normal, MultivariateNormal
from spey.system.exceptions import InvalidInput

from strathisla.nuisance_parameters import FullNuisanceParameters


def test_full_nuisance_parameters_creation():
    """Test creation of FullNuisanceParameters model"""
    
    data = np.array([36, 33])
    signal_yields = np.array([12.0, 15.0])
    background_yields = np.array([50.0, 48.0])
    
    signal_covariance = np.array([[1.44, 0.5], [0.5, 2.25]])
    background_covariance = np.array([[144.0, 13.0], [13.0, 256.0]])
    data_covariance = np.array([[36.0, 0.0], [0.0, 33.0]])
    
    stat_model = FullNuisanceParameters(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        signal_covariance=signal_covariance,
        background_covariance=background_covariance,
        data_covariance=data_covariance
    )
    
    assert stat_model is not None, "Model creation failed"
    assert hasattr(stat_model, 'signal_yields'), "Missing signal_yields attribute"
    assert hasattr(stat_model, 'background_yields'), "Missing background_yields attribute"
    assert hasattr(stat_model, 'signal_covariance'), "Missing signal_covariance attribute"
    assert hasattr(stat_model, 'background_covariance'), "Missing background_covariance attribute"
    assert hasattr(stat_model, 'data_covariance'), "Missing data_covariance attribute"

    # check data is assigned correctly
    np.testing.assert_array_equal(stat_model.data, data)
    np.testing.assert_array_equal(stat_model.signal_yields, signal_yields)
    np.testing.assert_array_equal(stat_model.background_yields, background_yields)
    np.testing.assert_array_equal(stat_model.signal_covariance, signal_covariance)
    np.testing.assert_array_equal(stat_model.background_covariance, background_covariance)
    np.testing.assert_array_equal(stat_model.data_covariance, data_covariance)


def test_full_nuisance_parameters_input_validation():
    """Test input validation for FullNuisanceParameters"""
    
    data = np.array([20, 25])
    signal_yields = np.array([5.0, 8.0])
    background_yields = np.array([18.0, 22.0])
    
    signal_covariance = np.diag([0.25, 0.64])
    background_covariance = np.diag([9.0, 16.0])
    data_covariance = np.diag([20.0, 25.0])
    
    # Test with mismatched dimensions
    with pytest.raises((InvalidInput)):
        FullNuisanceParameters(
            signal_yields=np.array([5.0]),  # Wrong size
            background_yields=background_yields,
            data=data,
            signal_covariance=signal_covariance,
            background_covariance=background_covariance,
            data_covariance=data_covariance
        )


def test_spey_plugin_registration():
    """Test that the plugin is properly registered with spey"""
    
    backend = spey.get_backend("strathisla.full_nuisance_parameters")
    assert backend is not None, "Plugin not registered with spey"
    
    # Test that we can create a model through the backend
    data = np.array([20, 25])
    signal_yields = np.array([5.0, 8.0])
    background_yields = np.array([18.0, 22.0])
    
    signal_covariance = np.diag([0.25, 0.64])
    background_covariance = np.diag([9.0, 16.0])
    data_covariance = np.diag([20.0, 25.0])
    
    stat_model = backend(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        signal_covariance=signal_covariance,
        background_covariance=background_covariance,
        data_covariance=data_covariance
    )
    
    assert stat_model is not None, "Backend creation failed"
        

def test_full_nuisance_parameters_calculations():
    """Test calculations with FullNuisanceParameters"""
    
    data = np.array([20, 25])
    signal_yields = np.array([5.0, 8.0])
    background_yields = np.array([18.0, 22.0])
    
    signal_covariance = np.diag([0.25, 0.64])
    background_covariance = np.diag([9.0, 16.0])
    data_covariance = np.diag([20.0, 25.0])
    
    backend = spey.get_backend("strathisla.full_nuisance_parameters")
    stat_model = backend(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        signal_covariance=signal_covariance,
        background_covariance=background_covariance,
        data_covariance=data_covariance
    )
    
    cls_obs = stat_model.exclusion_confidence_level()
    assert isinstance(cls_obs[0], float), "CLs should return a float"
    assert 0 <= cls_obs[0] <= 1, "CLs should be between 0 and 1"

    poi_upper_limit = stat_model.poi_upper_limit()
    assert isinstance(poi_upper_limit, float), "Upper limit should return a float"
    assert poi_upper_limit >= 0, "Upper limit should be non-negative"


def test_readme_example():
    """Check the example in the readme works"""
    
    data = np.array([30, 35, 40])
    signal_yields = np.array([8.0, 10.0, 12.0])
    background_yields = np.array([25.0, 28.0, 32.0])
        
    signal_covariance = np.array([[1.0, 0.2, 0.1], 
                                    [0.2, 1.5, 0.3], 
                                    [0.1, 0.3, 2.0]])
        
    background_covariance = np.array([[16.0, 4.0, 2.0],
                                        [4.0, 25.0, 5.0],
                                        [2.0, 5.0, 36.0]])
        
    data_covariance = np.array([[30.0,2.0,0.5], [4.3, 35.0, 9.0] , [6.2, 13.0, 40.0]])
    
    # check matrices are positive definite
    assert np.all(np.linalg.eigvals(signal_covariance) > 0), "Signal covariance not positive definite"
    assert np.all(np.linalg.eigvals(background_covariance) > 0), "Background covariance not positive definite"
    assert np.all(np.linalg.eigvals(data_covariance) > 0), "Data covariance not positive definite"
    
    backend = spey.get_backend("strathisla.full_nuisance_parameters")

    stat_model = backend(
        signal_yields=signal_yields,
        background_yields=background_yields,
        data=data,
        signal_covariance=signal_covariance,
        background_covariance=background_covariance,
        data_covariance=data_covariance
    )
    
    CLs = stat_model.exclusion_confidence_level()

    assert np.isclose(CLs[0],0.7583), "Result of example computation has changed"


def test_model_types():
    """Test the main and constraint model types for two bins"""
    two_bins = np.ones(2)
    two_bins_cov = np.identity(2)
    stat_model = FullNuisanceParameters(two_bins,two_bins,two_bins,
                         two_bins_cov,two_bins_cov,two_bins_cov)
    assert stat_model.main_model.pdf_type=='poiss'
    assert all([isinstance(model,MultivariateNormal) for model in stat_model.constraint_model._pdfs])

def test_one_bin_model_types():
    """Test the main and constraint model types for one bin"""
    one_bin = np.array([1.0])
    stat_model = FullNuisanceParameters(one_bin,one_bin,one_bin,one_bin,one_bin,one_bin)
    assert stat_model.main_model.pdf_type=='poiss'
    assert all([isinstance(model,Normal) for model in stat_model.constraint_model._pdfs])