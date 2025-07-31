"""Spey plugin for the likelihood described in arXiv:2102.04377"""

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

# pylint: disable=E1101,E1120
log = logging.getLogger("Spey")

# pylint: disable=W1203



class FullNuisanceParameters(BackendBase):
    r"""
    Spey implementation for the likelihood for a histogram with correlated uncertainties on the signal, background and data.  
    Described in arXiv:2102.04377. See eq. 7.

    .. math::

        L(\mu, \theta) = 
        \prod_{i \in {\rm bins}} 
        {\rm Poiss}
        (n_i \vert \mu s_i+b_i + \sum_{j \in n,s,b}  \theta_i^{(j)} \sigma_i^{(j)})
        \prod_{j \in n,s,b} 
        {\rm Gauss}(\theta^{(j)}|0,\Sigma^{(j)}) 

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        signal_covariance (``np.ndarray``): signal covariance matrix (must be square)
        background_covariance (``np.ndarray``): background covariance matrix (must be square)
        data_covariance (``np.ndarray``): data covariance matrix (must be square)
    """

    name: str = "strathisla.full_nuisance_parameters"
    """Name of the backend"""
    version: str = "1.0.0"
    """Version of the backend"""
    author: str = "Joe Egan (joe.egan.23@ucl.ac.uk)"
    """Author of the backend"""
    spey_requires: str = ">=0.0.1"
    """Spey version required for the backend"""
    doi: str = "10.21468/SciPostPhysCore.4.2.013"
    """Citable DOI for the backend"""
    arXiv: str = "2102.04377"
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        signal_covariance: np.ndarray,
        background_covariance: np.ndarray,
        data_covariance: np.ndarray
    ):  
        # need numpy arrays for the checks
        signal_yields = np.array(signal_yields)
        background_yields = np.array(background_yields)
        data = np.array(data)
        signal_covariance = np.array(signal_covariance)
        background_covariance = np.array(background_covariance)
        data_covariance = np.array(data_covariance)

        for np_arr in [signal_yields,background_yields,data,signal_covariance,background_covariance,data_covariance]:
            # check for single bin histo not passed as list, which results in an empty tuple for the .shape attribute
            if np_arr.shape == tuple():
                raise InvalidInput('Pass input arguments as lists or numpy arrays')
            # check for empty inputs
            if np_arr.shape[0] == 0:
                raise InvalidInput('Inputs must not be empty')

        # check all input yields have the same length
        if len(set((len(yields) for yields in (signal_yields,background_yields,data)))) != 1:
            raise InvalidInput('Yields must be the same length')
        
        for cov in (data_covariance,signal_covariance,background_covariance):
            # check input yields and covariance lengths match
            if len(data) != cov.shape[0]:
                raise InvalidInput('Covariance matrices size should match the number of yields')

            if len(data) > 1:
                # check all covariance matrices are 2D and square
                if cov.ndim != 2:
                    raise InvalidInput('2D covariance matrix required')
                if cov.shape[0] != cov.shape[1]:
                    raise InvalidInput('Covariance matrix must be square')

        # can assign these now they've been checked
        self.signal_yields = signal_yields
        self.background_yields = background_yields
        self.data = data

        self.signal_covariance = signal_covariance
        self.background_covariance = background_covariance
        self.data_covariance = data_covariance

        # set flag for single bin likelihood
        # also get uncertainties required for computation of Poisson mean in main model
        if len(self.data) == 1:
            self.single_bin = True
            self.signal_uncertainties = np.sqrt(self.signal_covariance)
            self.background_uncertainties = np.sqrt(self.background_covariance)
            self.data_uncertainties = np.sqrt(self.data_covariance)
        else:
            self.single_bin = False
            self.signal_uncertainties = np.sqrt(self.signal_covariance.diagonal())
            self.background_uncertainties = np.sqrt(self.background_covariance.diagonal())
            self.data_uncertainties = np.sqrt(self.data_covariance.diagonal())

        minimum_poi = -np.inf
        if self.is_alive:
            minimum_poi = -np.min(
                self.background_yields[self.signal_yields > 0.0]
                / self.signal_yields[self.signal_yields > 0.0]
            )
        log.debug(f"Min POI set to : {minimum_poi}")

        self._main_model = None
        self._constraint_model = None
        self.constraints = []
        """Constraints to be used during optimisation process"""

        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[1.0] * (3*len(data) + 1),
            suggested_bounds=[(minimum_poi, 10)]
            + [(None, None)] * 3*len(data),
        )

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return np.any(self.signal_yields > 0.0)

    def config(self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0
    ) -> ModelConfig:
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``10.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        if allow_negative_signal and poi_upper_bound == 10.0:
            return self._config

        return ModelConfig(
            self._config.poi_index,
            self._config.minimum_poi,
            self._config.suggested_init,
            [(0, poi_upper_bound)] + self._config.suggested_bounds[1:],
        )
    
    @property
    def constraint_model(self) -> ConstraintModel:
        """retreive constraint model distribution"""
        # set this so that the model for each nuisance source can only access its own nuisance parameters
        slice_start_index = (1,2,3)
        if self._constraint_model is None:
            if self.single_bin:
                # get a Gaussian with mean zero and standard deviation from the covariance matrix
                pdf_descs = [ 
                {
                    "distribution_type": "normal",
                    "args": [np.zeros(1), np.sqrt(cov)],
                    "kwargs": {"domain": slice(index, None, 3)},
                }
                # want a constraint pdf for each source of uncertainty
                for cov, index in zip((self.signal_covariance,self.background_covariance,self.data_covariance),slice_start_index)
                ]
                
            else:
                pdf_descs = [ 
                {
                    "distribution_type": "multivariatenormal",
                    "args": [np.zeros(len(self.data)), covariance_to_correlation(cov)],
                    "kwargs": {"domain": slice(index, None, 3)},
                }
                # want a constraint pdf for each source of uncertainty
                for cov, index in zip((self.signal_covariance,self.background_covariance,self.data_covariance), slice_start_index)
                ]

            self._constraint_model = ConstraintModel(pdf_descs)
        return self._constraint_model

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
                    expectation value of the poisson distribution with respect to
                    nuisance parameters.
                """
                # have 3 nuisance parameters for each bin, so 3N+1 in total for N bins
                # split the non-poi parameters into 3 seperate arrays for signal, background and data uncertainties
                signal_pars =  pars[slice(1,None,3)]
                background_pars = pars[slice(2,None,3)]
                data_pars = pars[slice(3,None,3)]
                return pars[0] * self.signal_yields + self.background_yields + signal_pars*self.signal_uncertainties + background_pars*self.background_uncertainties + data_pars*self.data_uncertainties

            def constraint(pars: np.ndarray) -> np.ndarray:
                """Compute constraint term"""
                signal_pars = pars[slice(1,None,3)]
                background_pars = pars[slice(2,None,3)]
                data_pars = pars[slice(3,None,3)]
                return signal_pars*self.signal_uncertainties + background_pars*self.background_uncertainties + data_pars*self.data_uncertainties

            jac_constr = jacobian(constraint)

            self.constraints.append(
                NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
            )

            self._main_model = MainModel(lam)

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
            ) - self.constraint_model.log_prob(pars)

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
        ) + self.constraint_model.log_prob(pars)

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
            ) + self.constraint_model.log_prob(pars)

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

        def sampler(sample_size: int, include_auxiliary: bool = True) -> np.ndarray:
            """
            Fucntion to generate samples.

            Args:
                sample_size (``int``): number of samples to be generated.
                include_auxiliary (``bool``): wether or not to include auxiliary data
                    coming from the constraint model.

            Returns:
                ``np.ndarray``:
                generated samples
            """
            sample = self.main_model.sample(pars, sample_size)

            if include_auxiliary:
                constraint_sample = self.constraint_model.sample(pars[1:], sample_size)
                sample = np.hstack([sample, constraint_sample])

            return sample

        return sampler

    def expected_data(
        self, pars: List[float], include_auxiliary: bool = True
    ) -> List[float]:
        r"""
        Compute the expected value of the statistical model

        Args:
            pars (``List[float]``): nuisance, :math:`\theta` and parameter of interest,
            :math:`\mu`.
            include_auxiliary (``bool``): wether or not to include auxiliary data
            coming from the constraint model.

        Returns:
            ``List[float]``:
            Expected data of the statistical model
        """
        data = self.main_model.expected_data(pars)

        if include_auxiliary:
            data = np.hstack([data, self.constraint_model.expected_data()])
        return data
