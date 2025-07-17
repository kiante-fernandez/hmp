"""Markov Chain Monte Carlo estimator for HMP models."""

import numpy as np
import warnings
from typing import Optional, Dict, Any

try:
    from .base import BaseEstimator, EstimationResult
except ImportError:
    # Fallback for direct imports
    from base import BaseEstimator, EstimationResult
from hmp.trialdata import TrialData

try:
    import pymc as pm
    import arviz as az
    import pytensor
    import pytensor.tensor as at
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None
    pytensor = None
    at = None


class MCMCEstimator(BaseEstimator):
    """MCMC parameter estimator using PyMC.
    
    This estimator provides Bayesian parameter estimation for HMP models
    using Markov Chain Monte Carlo sampling. It provides uncertainty
    quantification through posterior distributions.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of MCMC samples to draw. Default is 2000.
    n_tune : int, optional
        Number of tuning samples. Default is 1000.
    n_chains : int, optional
        Number of parallel chains. Default is 4.
    target_accept : float, optional
        Target acceptance rate for NUTS. Default is 0.8.
    random_seed : int, optional
        Random seed for reproducibility. Default is None.
    prior_config : dict, optional
        Prior configuration for parameters. Default uses weakly informative priors.
    """
    
    def __init__(self, n_samples: int = 2000, n_tune: int = 1000, n_chains: int = 4,
                 target_accept: float = 0.8, random_seed: Optional[int] = None,
                 prior_config: Optional[Dict[str, Any]] = None, **kwargs):
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is required for MCMCEstimator. Install with: pip install pymc")
            
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.prior_config = prior_config or {}
        self.trace = None
        self.model = None
        
    def fit(self, trial_data, initial_channel_pars: np.ndarray, 
           initial_time_pars: np.ndarray, model=None, **kwargs) -> EstimationResult:
        """Fit model parameters using MCMC.
        
        Parameters
        ----------
        trial_data : TrialData
            The trial data to fit the model to
        initial_channel_pars : np.ndarray
            Initial channel parameter values (used for initialization)
        initial_time_pars : np.ndarray
            Initial time distribution parameter values (used for initialization)
        model : EventModel
            The model instance (needed for likelihood computation)
        **kwargs
            Additional fitting options (currently unused)
            
        Returns
        -------
        EstimationResult
            Results of the MCMC estimation with uncertainty estimates
        """
        if model is None:
            raise ValueError("MCMCEstimator requires a model instance for likelihood computation")
        if not isinstance(trial_data, TrialData):
            trial_data = TrialData.from_preprocessed(trial_data, model.template)
            
        if initial_time_pars.ndim == 3:
            initial_time_pars = initial_time_pars[0]
        if initial_channel_pars.ndim == 3:
            initial_channel_pars = initial_channel_pars[0]
        n_events = initial_channel_pars.shape[-2]
        n_dims = initial_channel_pars.shape[-1]
        n_stages = initial_time_pars.shape[-2]
        
        # Build the PyMC model
        with pm.Model() as pymc_model:
            self.model = pymc_model
            
            # Channel parameter priors
            channel_pars = pm.Normal(
                "channel_pars", 
                mu=self.prior_config.get("channel_mu", 0.0),
                sigma=self.prior_config.get("channel_sigma", 1.0),
                shape=(n_events, n_dims),
                initval=initial_channel_pars
            )
            
            # Time parameter priors
            # Shape parameters (must be positive)
            shape_pars = pm.HalfNormal(
                "shape_pars",
                sigma=self.prior_config.get("shape_sigma", 2.0),
                shape=n_stages,
                initval=initial_time_pars[..., 0]
            )
            
            # Scale parameters (must be positive)
            scale_pars = pm.HalfNormal(
                "scale_pars", 
                sigma=self.prior_config.get("scale_sigma", 5.0),
                shape=n_stages,
                initval=initial_time_pars[..., 1]
            )
            
            # Combine time parameters
            time_pars = pm.Deterministic(
                "time_pars",
                pm.math.stack([shape_pars, scale_pars], axis=-1)
            )
            
            # Combine channel and time parameters into a single flattened variable
            flat_pars = pm.Deterministic(
                "flat_pars",
                pm.math.concatenate([channel_pars.flatten(), time_pars.flatten()])
            )

            # Custom likelihood based on HMP forward-backward algorithm
            likelihood = pm.Potential(
                "likelihood",
                self._log_likelihood_func(trial_data, model, channel_pars, time_pars)
            )
            
            # Sample from posterior
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                self.trace = pm.sample(
                    draws=self.n_samples,
                    tune=self.n_tune,
                    chains=self.n_chains,
                    random_seed=self.random_seed,
                    return_inferencedata=True,
                    progressbar=True
                )
        
        # Extract results
        posterior = self.trace.posterior
        
        # Point estimates (posterior means)
        channel_pars_mean = posterior["channel_pars"].mean(dim=["chain", "draw"]).values
        time_pars_mean = posterior["time_pars"].mean(dim=["chain", "draw"]).values
        
        # Compute final likelihood using the posterior means
        final_likelihood, _ = model.estim_probs(trial_data, channel_pars_mean, time_pars_mean)
        
        # Convergence diagnostics
        rhat = az.rhat(self.trace)
        ess = az.ess(self.trace)
        
        # Check convergence
        converged = bool(
            (rhat["channel_pars"].max() < 1.1).values and
            (rhat["shape_pars"].max() < 1.1).values and 
            (rhat["scale_pars"].max() < 1.1).values
        )
        
        # Uncertainty estimates
        uncertainty = {
            "channel_pars_std": posterior["channel_pars"].std(dim=["chain", "draw"]).values,
            "time_pars_std": posterior["time_pars"].std(dim=["chain", "draw"]).values,
            "channel_pars_quantiles": {
                "5%": posterior["channel_pars"].quantile(0.05, dim=["chain", "draw"]).values,
                "95%": posterior["channel_pars"].quantile(0.95, dim=["chain", "draw"]).values,
            },
            "time_pars_quantiles": {
                "5%": posterior["time_pars"].quantile(0.05, dim=["chain", "draw"]).values,
                "95%": posterior["time_pars"].quantile(0.95, dim=["chain", "draw"]).values,
            },
            "posterior_samples": {
                "channel_pars": posterior["channel_pars"].values,
                "time_pars": posterior["time_pars"].values,
            }
        }
        
        # Diagnostics
        diagnostics = {
            "method": "MCMC",
            "rhat": {
                "channel_pars": rhat["channel_pars"].values,
                "shape_pars": rhat["shape_pars"].values,
                "scale_pars": rhat["scale_pars"].values,
            },
            "ess": {
                "channel_pars": ess["channel_pars"].values,
                "shape_pars": ess["shape_pars"].values,
                "scale_pars": ess["scale_pars"].values,
            },
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "trace": self.trace
        }
        
        self._fitted = True
        
        return EstimationResult(
            channel_pars=channel_pars_mean,
            time_pars=time_pars_mean,
            likelihood=final_likelihood,
            converged=converged,
            n_iterations=self.n_samples,
            diagnostics=diagnostics,
            uncertainty=uncertainty
        )
    
    def _log_likelihood_func(self, trial_data, model, channel_pars, time_pars):
        """
        Returns a log-likelihood function for PyMC model.
        This function computes the HMP log-likelihood using the forward-backward algorithm.
        """
        import pytensor.tensor as at
        from pytensor.graph.op import Op
        from pytensor.graph.basic import Apply
        
        class HMPLogLikelihood(Op):
            """Custom PyTensor Op for HMP log-likelihood computation."""
            
            def make_node(self, channel_pars, time_pars):
                # Inputs
                channel_pars = at.as_tensor_variable(channel_pars)
                time_pars = at.as_tensor_variable(time_pars)
                # Output: scalar log-likelihood
                output = at.scalar('float64')
                return Apply(self, [channel_pars, time_pars], [output])
            
            def perform(self, node, inputs, outputs):
                channel_pars_np, time_pars_np = inputs
                
                # Compute log-likelihood using the HMP model's estim_probs
                log_likelihood, _ = model.estim_probs(trial_data, channel_pars_np, time_pars_np)
                outputs[0][0] = np.array(log_likelihood, dtype=np.float64)
        
        # Create and use the custom op
        hmp_likelihood_op = HMPLogLikelihood()
        likelihood_result = hmp_likelihood_op(channel_pars, time_pars)
        return likelihood_result

    def supports_uncertainty(self) -> bool:
        """MCMC provides uncertainty estimates through posterior distributions."""
        return True
        
    def get_posterior_samples(self, parameter: str = None) -> np.ndarray:
        """Get posterior samples for parameters.
        
        Parameters
        ----------
        parameter : str, optional
            Parameter name ('channel_pars', 'time_pars', or None for all)
            
        Returns
        -------
        np.ndarray or dict
            Posterior samples
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before getting posterior samples")
            
        if parameter is None:
            return {
                "channel_pars": self.trace.posterior["channel_pars"].values,
                "time_pars": self.trace.posterior["time_pars"].values
            }
        elif parameter in self.trace.posterior:
            return self.trace.posterior[parameter].values
        else:
            raise ValueError(f"Parameter '{parameter}' not found in trace")
    
    def plot_trace(self, parameter: str = None):
        """Plot MCMC traces for diagnostics.
        
        Parameters
        ----------
        parameter : str, optional
            Parameter to plot. If None, plots all parameters.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before plotting traces")
            
        if parameter is None:
            az.plot_trace(self.trace)
        else:
            az.plot_trace(self.trace, var_names=[parameter])
            
    def summary(self) -> str:
        """Get summary statistics of posterior distributions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before getting summary")
            
        return str(az.summary(self.trace))