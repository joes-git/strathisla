"""Spey implementation for the full Contur likelihood described in arXiv:2102.04377"""

from scipy.optimize import NonlinearConstraint
import numpy as np
from autograd import jacobian, grad

from spey import BackendBase, ExpectationType
from spey.base.model_config import ModelConfig
from spey.backends.distributions import ConstraintModel, MainModel
from spey.helper_functions import covariance_to_correlation



class ConturLikelihood(BackendBase):
    r"""
    Spey implementation for the likelihood described in arXiv:2102.04377. See eq. 7.

    .. math::

        L(\mu, \theta) = \prod_{i \in {\rm bins}} 
        {\rm Poiss} ( n_i \vert \mu s_i+b_i + \sum_{j \in n,s,b} \theta^{(j)}_i \sigma^{(j)}_i)
        \cdot
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

    name: str = "contur.full_likelihood"
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
        # check all input yields have the same length
        if len(set((len(yields) for yields in (signal_yields,background_yields,data)))) != 1:
            raise InvalidInput('Arrays of yields must be the same length')

        # check all covariance matrices are 2D and square
        for cov in (data_covariance,signal_covariance,background_covariance):
            if cov.ndim != 2:
                raise InvalidInput('2D covariance matrix required')
            if cov.shape[0] != cov.shape[1]:
                raise InvalidInput('Covariance matrix must be square')

        # check input yields and covariance lengths match
        if len(data) != data_covariance.shape[0]:
            raise InvalidInput('Covariance matrices size should match the number of yields')

        # can assign these now they've been checked
        self.signal_yields = signal_yields
        self.background_yields = background_yields
        self.data = data

        self.signal_covariance = signal_covariance
        self.background_covariance = background_covariance
        self.data_covariance = data_covariance

        minimum_poi = -np.inf
        if self.is_alive:
            minimum_poi = -np.min(
                self.background_yields[self.signal_yields > 0.0]
                / self.signal_yields[self.signal_yields > 0.0]
            )
        log.debug(f"Min POI set to : {minimum_poi}")

        self._config = ModelConfig(
            poi_index=0,
            minimum_poi=minimum_poi,
            suggested_init=[1.0] * (len(data) + 1)
            + (signal_uncertainty_configuration is not None)
            * ([1.0] * len(signal_yields)),
            suggested_bounds=[(minimum_poi, 10)]
            + [(None, None)] * len(data)
            + (signal_uncertainty_configuration is not None)
            * ([(None, None)] * len(signal_yields)),
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
        if self._constraint_model is None:
            # make a pdf description for each source of uncertainty
            pdf_descs = [ 
                {
                    "distribution_type": "multivariatenormal",
                    "args": [np.zeros(len(self.data)), covariance_to_correlation(cov)],
                    "kwargs": {"domain": slice(1, None)},
                }
                for cov in (self.signal_covariance,self.background_covariance,self.data_covariance)
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
                poisson_counts = (pars[0] * self.signal_yields + self.background_yields)
                # have 3 nuisance parameters for each bin, so 3N+1 in total for N bins
                # split the non-poi parameters into 3 seperate arrays for signal, background and data uncertainties
                signal_pars, background_pars, data_pars = np.array_split(pars[1:],3)

                signal_uncertainties = np.sqrt(self.signal_covariance.diagonal())
                background_uncertainties = np.sqrt(self.background_covariance.diagonal())
                data_uncertainties = np.sqrt(self.data_covariance.diagonal())

                return poisson_counts + signal_pars*signal_uncertainties + background_pars*background_uncertainties + data_pars*data_uncertainties

            def constraint(pars: np.ndarray) -> np.ndarray:
                """Compute constraint term"""
                # split the non-poi parameters into 3 seperate arrays for signal, background and data uncertainties
                signal_pars, background_pars, data_pars = np.array_split(pars[1:],3)

                signal_uncertainties = np.sqrt(self.signal_covariance.diagonal())
                background_uncertainties = np.sqrt(self.background_covariance.diagonal())
                data_uncertainties = np.sqrt(self.data_covariance.diagonal())

                return signal_pars*signal_uncertainties + background_pars*background_uncertainties + data_pars*data_uncertainties

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