"""Adaptation of the Multivariate Gaussian plugin, with both intereference and squared terms"""

import logging
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

from autograd import value_and_grad, hessian, jacobian
from autograd import numpy as np
from scipy.optimize import NonlinearConstraint

from spey._version import __version__
from spey.backends.distributions import ConstraintModel, MainModel
from spey.base import BackendBase, ModelConfig
from spey.helper_functions import covariance_to_correlation
from spey.optimizer import fit
from spey.utils import ExpectationType
from spey.system.exceptions import InvalidInput

# override for MainModel
from strathisla.variable_covariance import VariableCovMainModel

log = logging.getLogger("Spey")

class SimpleMultivariateGaussianEFT(BackendBase):
    r"""
    Multivariate Gaussian likelihood, with separated contributions from BSM+SM (interference) and pure BSM (squared) terms.
    This model has no nuisance parameters.

    .. math::
        \mathcal{L}(\mu) = \frac{1}{\sqrt{(2\pi)^k {\rm det}[\Sigma] }}
        \exp\left[-\frac{1}{2} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n)\Sigma^{-1} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n)^T \right]


    Args:
        quadratic_term (``np.ndarray``): signal that scales quadtratically with the parameter of interest
        linear_term (``np.ndarray``): signal that scales linearly with the parameter of interest
        background (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        covariance_matrix (``np.ndarray``): covariance matrix (must be square)
    """

    name: str = "strathisla.simple_multivariate_gaussian_eft"
    """Name of the backend"""
    version: str = "1.0.0"
    """Version of the backend"""
    author: str = "Joe Egan (joe.egan.23@ucl.ac.uk)"
    """Author of the backend"""
    spey_requires: str = ">=0.1.11"
    """Spey version required for the backend"""
    doi: str = ""
    """Citable DOI for the backend"""
    arXiv: str = ""
    """arXiv reference for the backend"""

    def __init__(
        self,
        quadratic_term: np.ndarray,
        linear_term: np.ndarray,
        background: np.ndarray,
        data: np.ndarray,
        covariance: np.ndarray
    ):  
        # need numpy arrays for the checks
        quadratic_term = np.array(quadratic_term)
        linear_term = np.array(linear_term)
        background = np.array(background)
        data = np.array(data)
        covariance = np.array(covariance)

        for np_arr in [quadratic_term, linear_term, background, data, covariance]: #[signal_yields,background_yields,data,signal_covariance,background_covariance,data_covariance]:
            # check for single bin histo not passed as list, which results in an empty tuple for the .shape attribute
            if np_arr.shape == tuple():
                raise InvalidInput('Pass input arguments as lists or numpy arrays')
            # check for empty inputs
            if np_arr.shape[0] == 0:
                raise InvalidInput('Inputs must not be empty')

        # check all input yields have the same length
        if len(set((len(yields) for yields in (quadratic_term, linear_term, background, data)))) != 1:
            raise InvalidInput('Input arrays must be the same length')
        
        # check input yields and covariance lengths match
        if len(data) != covariance.shape[0]:
            raise InvalidInput('Covariance matrix size should match the number of yields')

        if len(data) > 1:
            # check covariance matrix is 2D and square
            if covariance.ndim != 2:
                raise InvalidInput('2D covariance matrix required')
            if covariance.shape[0] != covariance.shape[1]:
                raise InvalidInput('Covariance matrix must be square')

        # can assign these now they've been checked
        self.quadratic_term = quadratic_term
        self.linear_term = linear_term
        self.background = background
        self.data = data
        self.covariance = covariance

        minimum_poi = -np.inf
        if self.is_alive:
            minimum_poi = -np.min(
                self.background[self.quadratic_term > 0.0]
                / self.quadratic_term[self.quadratic_term > 0.0]
            )
        log.debug(f"Min POI set to : {minimum_poi}")

        self._main_model = None
        self._constraint_model = None
        self.constraints = []

        """Constraints to be used during optimisation process"""
        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[0.0],
            suggested_bounds=[(minimum_poi, 50)]
        )

        self._main_kwargs = {
            "cov": self.covariance,
            "pdf_type": "multivariategauss",
        }

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return np.any(self.quadratic_term > 0.0) or np.any(self.linear_term > 0.0)

    def config(self, allow_negative_signal: bool = True, poi_upper_bound: float = 50.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``50.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        if allow_negative_signal and poi_upper_bound == 50.0:
            return self._config

        return ModelConfig(
            self._config.poi_index,
            self._config.minimum_poi,
            self._config.suggested_init,
            [(0, poi_upper_bound)] + self._config.suggested_bounds[1:],
        )
    
    @property
    def main_model(self) -> MainModel:
        """retreive the main model distribution"""
        if self._main_model is None:

            def lam(pars: np.ndarray) -> np.ndarray:
                """
                Compute lambda for Main model.

                Args:
                    pars (``np.ndarray``): nuisance parameters

                Returns:
                    ``np.ndarray``:
                    expectation value of the poisson distribution.
                """
                return pars[0]**2 * self.quadratic_term + pars[0] * self.linear_term + self.background

            self._main_model = MainModel(lam, **self._main_kwargs)

        return self._main_model

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
        r"""
        Objective function i.e. twice negative log-likelihood, :math:`-2\log\mathcal{L}(\mu, \theta)`

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
            p-values to be computed.

            * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
            * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
            * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.ndarray``, default ``None``): input data that to fit
            do_grad (``bool``, default ``True``): If ``True`` return objective and its gradient
            as ``tuple`` if ``False`` only returns objective function.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            Function which takes fit parameters (:math:`\mu` and :math:`\theta`) and returns either
            objective or objective and its gradient.
        """
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
            """Compute twice negative log-likelihood"""
            return -self.main_model.log_prob(
                pars, data[: len(self.data)]
            )

        if do_grad:
            return value_and_grad(negative_loglikelihood, argnum=0)

        return negative_loglikelihood

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.array] = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        r"""
        Generate function to compute :math:`\log\mathcal{L}(\mu, \theta)` where :math:`\mu` is the
        parameter of interest and :math:`\theta` are nuisance parameters.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
            p-values to be computed.

            * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
            * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
            * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.array``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and computes
            :math:`\log\mathcal{L}(\mu, \theta)`.
        """
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        return lambda pars: self.main_model.log_prob(
            pars, data[: len(self.data)]
        )

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Currently Hessian of :math:`\log\mathcal{L}(\mu, \theta)` is only used to compute
        variance on :math:`\mu`. This method returns a callable function which takes fit
        parameters (:math:`\mu` and :math:`\theta`) and returns Hessian.

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
            p-values to be computed.

            * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
            * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
            * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.ndarray``, default ``None``): input data that to fit

        Returns:
            ``Callable[[np.ndarray], float]``:
            Function that takes fit parameters (:math:`\mu` and :math:`\theta`) and
            returns Hessian of :math:`\log\mathcal{L}(\mu, \theta)`.
        """
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        def log_prob(pars: np.ndarray) -> np.ndarray:
            """Compute log-probability"""
            return self.main_model.log_prob(
                pars, data[: len(self.data)]
            )

        return hessian(log_prob, argnum=0)

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Retreives the function to sample from.

        Args:
            pars (``np.ndarray``): fit parameters (:math:`\mu` and :math:`\theta`)
            include_auxiliary (``bool``): wether or not to include auxiliary data
            coming from the constraint model.

        Returns:
            ``Callable[[int, bool], np.ndarray]``:
            Function that takes ``number_of_samples`` as input and draws as many samples
            from the statistical model.
        """

        def sampler(sample_size: int) -> np.ndarray:
            """
            Fucntion to generate samples.

            Args:
                sample_size (``int``): number of samples to be generated.

            Returns:
                ``np.ndarray``:
                generated samples
            """
            sample = self.main_model.sample(pars, sample_size)

            return sample

        return sampler

    def expected_data(
        self, pars: List[float]
    ) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,
            :math:`\mu`.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        data = self.main_model.expected_data(pars)

        return data

class MultivariateGaussianCovarianceScaledEFT(BackendBase):
    r"""
    Multivariate Gaussian likelihood, with seperated contributions from BSM+SM (interference) and pure BSM (squared) terms.
    The covariance matrix is scaled by the parameter of interest, :math:`\mu`.

    .. math::
         \mathcal{L}(\mu) = \frac{1}{\sqrt{(2\pi)^k {\rm det}[\Sigma(\mu)] }}
        \exp\left[-\frac{1}{2} (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n)\Sigma^{-1}(\mu) (\mu^2 s_{\text{quad}} + \mu s_{\text{lin}} + b - n)^T \right]


    Args:
        quadratic_term (``np.ndarray``): contribution from pure BSM diagrams
        linear_term (``np.ndarray``): 
        background (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        covariance_matrix (``np.ndarray``): covariance matrix (must be square)
    """

    name: str = "strathisla.multivariate_gaussian_scaled_covariance_eft"
    """Name of the backend"""
    version: str = "1.0.0"
    """Version of the backend"""
    author: str = "Joe Egan (joe.egan.23@ucl.ac.uk)"
    """Author of the backend"""
    spey_requires: str = ">=0.1.11"
    """Spey version required for the backend"""
    doi: str = ""
    """Citable DOI for the backend"""
    arXiv: str = ""
    """arXiv reference for the backend"""

    def __init__(
        self,
        quadratic_term: np.ndarray,
        linear_term: np.ndarray,
        background: np.ndarray,
        data: np.ndarray,
        data_covariance: np.ndarray,
        background_covariance: np.ndarray,
        quadratic_term_covariance: np.ndarray,
        linear_term_covariance: np.ndarray,
    ):  
        # need numpy arrays for the checks
        quadratic_term = np.array(quadratic_term)
        linear_term = np.array(linear_term)
        background = np.array(background)
        data = np.array(data)
        data_covariance = np.array(data_covariance)
        background_covariance = np.array(background_covariance)
        quadratic_term_covariance = np.array(quadratic_term_covariance)
        linear_term_covariance = np.array(linear_term_covariance)

        # check all input yields have the same length
        if len(set((len(yields) for yields in (quadratic_term, linear_term, background, data)))) != 1:
            raise InvalidInput('Input arrays must be the same length')
        
        # check input yields and covariance lengths match
        if len(data) != data_covariance.shape[0]:
            raise InvalidInput('Covariance matrix size should match the number of yields')

        # can assign these now they've been checked
        self.quadratic_term = quadratic_term
        self.linear_term = linear_term
        self.background = background
        self.data = data
        self.data_covariance = data_covariance
        self.background_covariance = background_covariance
        self.quadratic_term_covariance = quadratic_term_covariance
        self.linear_term_covariance = linear_term_covariance

        minimum_poi = -np.inf
        if self.is_alive:
            minimum_poi = -np.min(
                self.background[self.quadratic_term > 0.0]
                / self.quadratic_term[self.quadratic_term > 0.0]
            )
        log.debug(f"Min POI set to : {minimum_poi}")

        self._main_model = None
        self._constraint_model = None
        self.constraints = []

        """Constraints to be used during optimisation process"""
        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[0.0],
            suggested_bounds=[(minimum_poi, 50)]
        )

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return np.any(self.quadratic_term > 0.0) or np.any(self.linear_term > 0.0)

    def config(self, allow_negative_signal: bool = True, poi_upper_bound: float = 50.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``50.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        if allow_negative_signal and poi_upper_bound == 50.0:
            return self._config

        return ModelConfig(
            self._config.poi_index,
            self._config.minimum_poi,
            self._config.suggested_init,
            [(0, poi_upper_bound)] + self._config.suggested_bounds[1:],
        )
    
    @property
    def main_model(self) -> VariableCovMainModel:
        """retreive the main model distribution"""
        if self._main_model is None:

            def lam(pars: np.ndarray) -> np.ndarray:
                """
                Compute lambda for Main model.

                Args:
                    pars (``np.ndarray``): nuisance parameters

                Returns:
                    ``np.ndarray``:
                    expectation value of the poisson distribution.
                """
                return pars[0]**2 * self.quadratic_term + pars[0] * self.linear_term + self.background

            def cov(pars: np.ndarray) -> np.ndarray:
                """
                Covariance matrix scaled with the parameter of interest.

                Args:
                    pars (``np.ndarray``): nuisance parameters

                Returns:
                    ``np.ndarray``:
                    covariance matrix of the distribution.
                """
                return self.data_covariance + self.background_covariance + pars[0]**2 * self.linear_term_covariance + pars[0]**4 * self.quadratic_term_covariance

            self._main_model = VariableCovMainModel(lam, cov, pdf_type="multivariategauss")

        return self._main_model

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
        do_grad: bool = True,
    ) -> Callable[[np.ndarray], Union[Tuple[float, np.ndarray], float]]:
        r"""
        Objective function i.e. twice negative log-likelihood, :math:`-2\log\mathcal{L}(\mu, \theta)`

        Args:
            expected (~spey.ExpectationType): Sets which values the fitting algorithm should focus and
            p-values to be computed.

            * :obj:`~spey.ExpectationType.observed`: Computes the p-values with via post-fit
                prescriotion which means that the experimental data will be assumed to be the truth
                (default).
            * :obj:`~spey.ExpectationType.aposteriori`: Computes the expected p-values with via
                post-fit prescriotion which means that the experimental data will be assumed to be
                the truth.
            * :obj:`~spey.ExpectationType.apriori`: Computes the expected p-values with via pre-fit
                prescription which means that the SM will be assumed to be the truth.
            data (``np.ndarray``, default ``None``): input data that to fit
            do_grad (``bool``, default ``True``): If ``True`` return objective and its gradient
            as ``tuple`` if ``False`` only returns objective function.

        Returns:
            ``Callable[[np.ndarray], Union[float, Tuple[float, np.ndarray]]]``:
            Function which takes fit parameters (:math:`\mu` and :math:`\theta`) and returns either
            objective or objective and its gradient.
        """
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
            """Compute twice negative log-likelihood"""
            return -self.main_model.log_prob(
                pars, data[: len(self.data)]
            )

        if do_grad:
            return value_and_grad(negative_loglikelihood, argnum=0)

        return negative_loglikelihood

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.array] = None,
    ) -> Callable[[np.ndarray, np.ndarray], float]:
        r"""
        Generate function to compute :math:`\log\mathcal{L}(\mu, \theta)` where :math:`\mu` is the
        parameter of interest and :math:`\theta` are nuisance parameters.
        """
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        return lambda pars: self.main_model.log_prob(
            pars, data[: len(self.data)]
        )

    def get_hessian_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: Optional[np.ndarray] = None,
    ) -> Callable[[np.ndarray], float]:
        r"""
        Currently Hessian of :math:`\log\mathcal{L}(\mu, \theta)` is only used to compute
        variance on :math:`\mu`. This method returns a callable function which takes fit
        parameters (:math:`\mu` and :math:`\theta`) and returns Hessian.
        """
        current_data = (
            self.background_yields if expected == ExpectationType.apriori else self.data
        )
        data = current_data if data is None else data
        log.debug(f"Data: {data}")

        def log_prob(pars: np.ndarray) -> np.ndarray:
            """Compute log-probability"""
            return self.main_model.log_prob(
                pars, data[: len(self.data)]
            )

        return hessian(log_prob, argnum=0)

    def get_sampler(self, pars: np.ndarray) -> Callable[[int], np.ndarray]:
        r"""
        Retreives the function to sample from.

        Args:
            pars (``np.ndarray``): fit parameters (:math:`\mu` and :math:`\theta`)

        Returns:
            ``Callable[[int, bool], np.ndarray]``:
            Function that takes ``number_of_samples`` as input and draws as many samples
            from the statistical model.
        """

        def sampler(sample_size: int) -> np.ndarray:
            """
            Fucntion to generate samples.

            Args:
                sample_size (``int``): number of samples to be generated.

            Returns:
                ``np.ndarray``:
                generated samples
            """
            sample = self.main_model.sample(pars, sample_size)

            return sample

        return sampler

    def expected_data(
        self, pars: List[float]
    ) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,
            :math:`\mu`.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        data = self.main_model.expected_data(pars)

        return data