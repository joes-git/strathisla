"""Spey implementation for the likelihood described in arXiv:2007.08526"""

from scipy.optimize import NonlinearConstraint
import numpy as np

from spey import BackendBase, ExpectationType
from spey.base.model_config import ModelConfig
from spey.backends.distributions import ConstraintModel, MainModel
from autograd import jacobian, grad


class NeutrinoExperiment(BackendBase):
    r"""
    Spey implementation for the likelihood described in arXiv:2007.08526. See eq. A2.

    .. math::

        \mathcal{L}(\mu, \theta) = \left[\prod_{i\in{\rm channels}}\prod_{j_i\in {\rm bins}}
        {\rm Poiss}(n^j|(\mu n_s^j + n_b^j)(1 + \theta^j\sigma_b^j))\right] \cdot
        \prod_{k\in{\rm nuis}}\mathcal{N}(\theta^k | 0, 1)

    Args:
        signal_yields (``np.ndarray``): signal yields
        background_yields (``np.ndarray``): background yields
        data (``np.ndarray``): observations
        absolute_uncertainties (``np.ndarray``): absolute uncertainty values
    """

    name: str = "example.neutrino"
    """Name of the backend"""
    version: str = "1.0.0"
    """Version of the backend"""
    author: str = "SpeysideHEP"
    """Author of the backend"""
    spey_requires: str = ">=0.0.1"
    """Spey version required for the backend"""
    doi: str = "10.1103/PhysRevD.103.013004"
    """Citable DOI for the backend"""
    arXiv: str = "2007.08526"
    """arXiv reference for the backend"""

    def __init__(
        self,
        signal_yields: np.ndarray,
        background_yields: np.ndarray,
        data: np.ndarray,
        absolute_uncertainties: np.ndarray,
    ):
        self.signal_yields = np.array(signal_yields)
        self.background_yields = np.array(background_yields)
        self.data = np.array(data)

        def lam(pars: np.ndarray) -> np.ndarray:
            """
            Compute lambda for Main model with third moment expansion.
            For details see above eq 2.6 in :xref:`1809.05548`

            Args:
                pars (``np.ndarray``): nuisance parameters

            Returns:
                ``np.ndarray``:
                expectation value of the poisson distribution with respect to
                nuisance parameters.
            """
            return (pars[0] * self.signal_yields + self.background_yields) * (
                1 + pars[1:] * np.array(absolute_uncertainties)
            )

        def constraint(pars: np.ndarray) -> np.ndarray:
            """Compute constraint term"""
            return self.background_yields * (
                1 + pars[1:] * np.array(absolute_uncertainties)
            )

        jac_constr = jacobian(constraint)

        self.constraints = [
            NonlinearConstraint(constraint, 0.0, np.inf, jac=jac_constr)
        ]

        self.main_model = MainModel(lam)

        self.constraint_model: ConstraintModel = ConstraintModel(
            [
                {
                    "distribution_type": "normal",
                    "args": [
                        np.zeros(len(self.data)),
                        np.ones(len(self.data)),
                    ],
                    "kwargs": {"domain": slice(1, None)},
                }
            ]
        )

    @property
    def is_alive(self) -> bool:
        """Returns True if at least one bin has non-zero signal yield."""
        return np.any(self.signal_yields > 0.0)

    def config(self, allow_negative_signal: bool = True, poi_upper_bound: float = 10.0):
        r"""
        Model configuration.

        Args:
            allow_negative_signal (``bool``, default ``True``): If ``True`` :math:`\hat\mu`
              value will be allowed to be negative.
            poi_upper_bound (``float``, default ``40.0``): upper bound for parameter
              of interest, :math:`\mu`.

        Returns:
            ~spey.base.ModelConfig:
            Model configuration. Information regarding the position of POI in
            parameter list, suggested input and bounds.
        """
        min_poi = -np.min(
            self.background_yields[self.signal_yields > 0]
            / self.signal_yields[self.signal_yields > 0]
        )

        return ModelConfig(
            0,
            min_poi,
            [1.0] * (len(self.data) + 1),
            [(min_poi if allow_negative_signal else 0.0, poi_upper_bound)]
            + [
                (None, None),
            ]
            * len(self.data),
        )

    def get_objective_function(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: np.ndarray = None,
        do_grad: bool = True,
    ):
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

        def negative_loglikelihood(pars: np.ndarray) -> np.ndarray:
            """Compute twice negative log-likelihood"""
            return -self.main_model.log_prob(
                pars, data[: len(self.data)]
            ) - self.constraint_model.log_prob(pars)

        if do_grad:
            grad_negative_loglikelihood = grad(negative_loglikelihood, argnum=0)
            return lambda pars: (
                negative_loglikelihood(pars),
                grad_negative_loglikelihood(pars),
            )

        return negative_loglikelihood

    def get_logpdf_func(
        self,
        expected: ExpectationType = ExpectationType.observed,
        data: np.ndarray = None,
    ):
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

        return lambda pars: self.main_model.log_prob(
            pars, data[: len(self.data)]
        ) + self.constraint_model.log_prob(pars)

    def get_sampler(self, pars: np.ndarray):
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

    def expected_data(self, pars: list[float], include_auxiliary: bool = True):
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

        if include_auxiliary:
            data = np.hstack([data, self.constraint_model.expected_data()])
        return data
