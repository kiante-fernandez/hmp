"""Parameter estimation methods for HMP models."""

from .base import BaseEstimator, EstimationResult
from .em import EMEstimator
from .utils import (
    ConvergenceChecker, 
    RelativeLikelihoodConvergence,
    ParameterConvergence,
    compute_log_likelihood,
    validate_parameters,
    initialize_parameters
)

# Optional MCMC estimator (requires PyMC)
try:
    from .mcmc import MCMCEstimator
    MCMC_AVAILABLE = True
except ImportError:
    MCMCEstimator = None
    MCMC_AVAILABLE = False

__all__ = [
    "BaseEstimator", 
    "EstimationResult", 
    "EMEstimator",
    "ConvergenceChecker",
    "RelativeLikelihoodConvergence", 
    "ParameterConvergence",
    "compute_log_likelihood",
    "validate_parameters",
    "initialize_parameters"
]

if MCMC_AVAILABLE:
    __all__.append("MCMCEstimator")