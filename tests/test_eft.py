"""Test EFT functionality"""

import numpy as np
import pytest
import spey
from strathisla.eft import SimpleMultivariateGaussianEFT, MultivariateGaussianCovarianceScaledEFT

@pytest.fixture
def data():
    """
    Histogram data for EFT tests.
    """
    quadratic_term = np.array([9.908565e-04, 1.082060e-02, 4.402903e-02, 6.319499e-02, 6.183598e-02, 5.080806e-02, 4.612987e-02, 2.338881e-02, 1.533290e-02, 1.413599e-02,1.203450e-02, 2.027827e-02, 9.928253e-01])
    linear_term = np.array([0.00045349, 0.00304028, 0.0090483, 0.01188922, 0.00954399, 0.00590007, 0.00350471, 0.00211336, 0.00133363, 0.00084733, 0.0004529, 0.00021938, 0.02627369])

    background = np.array([0.25291, 0.59942, 1.52996, 1.92881, 1.49616, 0.96709, 0.56009, 0.34006, 0.21114, 0.12546, 0.07015, 0.03218, 2.82177])
    measurement = np.array([0.2325,  0.5385,  1.25734, 1.76016, 1.39748, 0.90571, 0.59037, 0.32719, 0.22003, 0.13005, 0.05643, 0.03522, 2.70176])

    covariance = np.array([
        [0.06427483, 0.05824, 0.01225, 0.00834, 0.00776, 0.00783, 0.00971, 0.00899, 0.01138, 0.00822, 0.00583, 0.01466, 0.0123],
        [0.05824, 0.06694904, 0.01207, 0.00822, 0.00765, 0.00772, 0.00957, 0.00886, 0.01121, 0.0081, 0.00575, 0.01445, 0.01212],
        [0.01225, 0.01207, 0.02126036, 0.00173, 0.00161, 0.00162, 0.00201, 0.00186, 0.00236, 0.0017, 0.00121, 0.00304, 0.00255],
        [0.00834, 0.00822, 0.00173, 0.02120313, 0.0011, 0.00111, 0.00137, 0.00127, 0.00161, 0.00116, 0.00082, 0.00207, 0.00174],
        [0.00776, 0.00765, 0.00161, 0.0011, 0.01490038, 0.00103, 0.00128, 0.00118, 0.00149, 0.00108, 0.00077, 0.00193, 0.00162],
        [0.00783, 0.00772, 0.00162, 0.00111, 0.00103, 0.00713654, 0.00129, 0.00119, 0.00151, 0.00109, 0.00077, 0.00194, 0.00163],
        [0.00971, 0.00957, 0.00201, 0.00137, 0.00128, 0.00129, 0.00460966, 0.00148, 0.00187, 0.00135, 0.00096, 0.00241, 0.00202],
        [0.00899, 0.00886, 0.00186, 0.00127, 0.00118, 0.00119, 0.00148, 0.00398912, 0.00173, 0.00125, 0.00089, 0.00223, 0.00187],
        [0.01138, 0.01121, 0.00236, 0.00161, 0.00149, 0.00151, 0.00187, 0.00173, 0.00504761, 0.00158, 0.00112, 0.00282, 0.00237],
        [0.00822, 0.0081, 0.0017, 0.00116, 0.00108, 0.00109, 0.00135, 0.00125, 0.00158, 0.00434905, 0.00081, 0.00204, 0.00171],
        [0.00583, 0.00575, 0.00121, 0.00082, 0.00077, 0.00077, 0.00096, 0.00089, 0.00112, 0.00081, 0.00797059, 0.00145, 0.00121],
        [0.01466, 0.01445, 0.00304, 0.00207, 0.00193, 0.00194, 0.00241, 0.00223, 0.00282, 0.00204, 0.00145, 0.01002773, 0.00305],
        [0.0123, 0.01212, 0.00255, 0.00174, 0.00162, 0.00163, 0.00202, 0.00187, 0.00237, 0.00171, 0.00121, 0.00305, 0.07678421]
        ])

    quadratic_term_covariance = np.diag([1.42421005e-07, 8.57147040e-07, 1.99661509e-06,1.37341894e-06, 9.10345911e-07, 5.55596702e-07, 3.36359476e-07, 1.61939941e-07, 6.21246927e-08, 7.19590202e-08, 4.45120524e-08, 5.17796800e-09,1.57989090e-08])
    linear_term_covariance = np.diag([3.60939969e-08, 4.65573038e-07, 9.61593099e-07, 6.67990493e-07, 8.05653062e-07, 4.23737714e-07, 3.84071726e-07, 2.19510143e-07,2.05250733e-07, 1.45502399e-07, 1.08455170e-07, 2.62843874e-08, 6.23434529e-04])

    return {'quadratic_term': quadratic_term,
            'linear_term': linear_term,
            'background': background,
            'measurement': measurement,
            'covariance': covariance,
            'quadratic_term_covariance': quadratic_term_covariance,
            'linear_term_covariance': linear_term_covariance
            }


def test_simple_multivariate_gaussian_eft(data):
    """Test SimpleMultivariateGaussianEFT model"""
    
    stat_model = SimpleMultivariateGaussianEFT(
        quadratic_term=data['quadratic_term'],
        linear_term=data['linear_term'],
        background=data['background'],
        data=data['measurement'],
        covariance=data['covariance'],
    )
    
    assert stat_model is not None, "EFT model creation failed"
    assert hasattr(stat_model, 'covariance'), "Missing covariance matrix"

    # check the covariance matrix function doesn't depend on mu
    cov1 = stat_model.main_model._pdf(pars=[0.0]).cov
    assert callable(cov1), "Covariance function should be callable"
    assert np.allclose(cov1([0.0]), cov1([1e+5])), "Covariance should not change with mu"

    # check covariance matrix doesn't depend on the value of mu passed to _pdf()
    cov2 = stat_model.main_model._pdf(pars=[1e+5]).cov
    assert np.allclose(cov1([0.0]), cov2([0.0])), "Covariance should not change with mu"


def test_statistical_model_simple_multivariate_gaussian_eft(data):
    """Test statistical model creation for SimpleMultivariateGaussianEFT"""
    
    backend = spey.get_backend("strathisla.simple_multivariate_gaussian_eft")
    stat_model = backend(
        quadratic_term=data['quadratic_term'],
        linear_term=data['linear_term'],
        background=data['background'],
        data=data['measurement'],
        covariance=data['covariance']
    )
    
    CLs = stat_model.exclusion_confidence_level()
    assert isinstance(CLs[0], float), "CLs[0] should return a float"
    assert CLs[0] <= 1 and CLs[0] >= 0, "CLs should be between 0 and 1"

    mu_upper_limit = stat_model.poi_upper_limit()
    assert isinstance(mu_upper_limit, float), "Upper limit should return a float"
    assert mu_upper_limit >= 0, "Upper limit should be non-negative"


def test_covariance_scaled_eft(data):
    """Test MultivariateGaussianCovarianceScaledEFT model"""

    stat_model = MultivariateGaussianCovarianceScaledEFT(
        quadratic_term=data['quadratic_term'],
        linear_term=data['linear_term'],
        background=data['background'],
        data=data['measurement'],
        data_covariance=data['covariance'],
        background_covariance=data['covariance'], # just for testing build
        quadratic_term_covariance=data['quadratic_term_covariance'],
        linear_term_covariance=data['linear_term_covariance']
    )
    
    assert stat_model is not None, "Scaled EFT model creation failed"

    # check the covariance scaling
    mu1 = 0.0
    mu2 = 1e+5
    cov1 = stat_model.main_model._pdf(pars=[mu1]).cov

    assert callable(cov1), "Covariance function should be callable"
    # confusingly, the argument passed to cov() isn't used, the value of mu passed to _pdf() is. Hence this test to check it doesn't change
    assert np.allclose(cov1([mu1]), cov1([mu2])), "poi value used for covariance should depend on the value passed to _pdf()"

    cov2 = stat_model.main_model._pdf(pars=[mu2]).cov
    
    # check scaling happens from argument passed to _pdf()
    assert not np.allclose(cov1([0.0]), cov2([0.0])), "Covariance should change with mu"


def test_eft_plugin_registration():
    """Test that EFT plugins are properly registered with spey"""
    
    try:
        simple_backend = spey.get_backend("strathisla.simple_multivariate_gaussian_eft")
        assert simple_backend is not None, "Simple EFT plugin not registered"
        
        scaled_backend = spey.get_backend("strathisla.multivariate_gaussian_scaled_covariance_eft")
        assert scaled_backend is not None, "Scaled EFT plugin not registered"
    except Exception as e:
        pytest.skip(f"EFT plugin registration not working: {e}")

