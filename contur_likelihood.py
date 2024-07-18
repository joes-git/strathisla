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
                # split the remaining parameters into 3 seperate arrays for signal, background and data uncertainties
                signal_pars, background_pars, data_pars = np.array_split(pars[1:],3)

                signal_uncertainties = np.sqrt(self.signal_covariance.diagonal())
                background_uncertainties = np.sqrt(self.background_covariance.diagonal())
                data_uncertainties = np.sqrt(self.data_covariance.diagonal())

                return poisson_counts + signal_pars*signal_uncertainties + background_pars*background_uncertainties + data_pars*data_uncertainties

            def constraint(pars: np.ndarray) -> np.ndarray:
                """Compute constraint term"""
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