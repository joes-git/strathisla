import numpy as np
from typing import Any, Callable, Dict, List, Literal, Union

from spey.backends.distributions import MainModel, MultivariateNormal, Normal

class VariableCovMainModel(MainModel):
    """
    Variant on the main model that allows for the covariance matrix to be a function of the nuisance parameters.

    Args:
        loc (``Callable[[np.ndarray], np.ndarray]``): callable function that represents
          equivalent to the expectation value of the Poisson distribution. It takes nuisance parameters as input.
        cov (``Callable[[np.ndarray], np.ndarray]``): covariance matrix of the distribution, it takes nuisance parameters as input 
    """

    def __init__(
        self,
        loc: Callable[[np.ndarray], np.ndarray],
        cov: Callable[[np.ndarray], np.ndarray],
        pdf_type: Literal["gauss", "multivariategauss"] = "multivariategauss",
    ):
        self.pdf_type = pdf_type
        """Type of the PDF"""
        if pdf_type == "multivariategauss" and cov is not None:
            self._pdf = lambda pars: MultivariateNormal(mean=loc(pars), cov=cov(pars))
        elif pdf_type == "gauss" and cov is not None:
            self._pdf = lambda pars: Normal(loc=loc(pars), scale=cov(pars))
        else:
            raise DistributionError("Unknown pdf type or associated input.")