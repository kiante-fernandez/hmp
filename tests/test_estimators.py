"""Tests for the parameter estimation architecture."""

import numpy as np
import pytest
import numpy as np

from hmp.estimators import BaseEstimator, EstimationResult, EMEstimator, MCMCEstimator
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.mcmc import PYMC_AVAILABLE
from hmp.estimators.utils import (
    validate_parameters, 
    initialize_parameters,
    RelativeLikelihoodConvergence,
    compute_log_likelihood
)
from hmp.models import EventModel
from hmp.preprocessing import Standard
from hmp.simulations import demo


class TestEstimationResult:
    """Test EstimationResult class."""
    
    def test_creation(self):
        """Test basic EstimationResult creation."""
        result = EstimationResult(
            channel_pars=np.random.randn(2, 5),
            time_pars=np.random.rand(3, 2) + 0.1,
            likelihood=-100.5,
            converged=True,
            n_iterations=25,
            diagnostics={"method": "test"}
        )
        
        assert result.likelihood == -100.5
        assert result.converged is True
        assert result.n_iterations == 25
        assert result.channel_pars.shape == (2, 5)
        assert result.time_pars.shape == (3, 2)
        assert result.diagnostics["method"] == "test"

    def test_array_conversion(self):
        """Test automatic array conversion."""
        result = EstimationResult(
            channel_pars=[[1.0, 2.0], [3.0, 4.0]],  # List input
            time_pars=[[2.0, 1.5], [2.0, 2.0]],     # List input
            likelihood=-50.0,
            converged=True,
            n_iterations=10,
            diagnostics={}
        )
        
        assert isinstance(result.channel_pars, np.ndarray)
        assert isinstance(result.time_pars, np.ndarray)
        assert result.channel_pars.shape == (2, 2)


class TestEMEstimator:
    """Test EMEstimator class."""
    
    def test_creation(self):
        """Test EMEstimator creation with parameters."""
        em_est = EMEstimator(max_iteration=500, tolerance=1e-5, min_iteration=5)
        
        assert em_est.max_iteration == 500
        assert em_est.tolerance == 1e-5
        assert em_est.min_iteration == 5
        assert em_est.get_method_name() == "EMEstimator"
        assert em_est.supports_uncertainty() is False

    def test_default_parameters(self):
        """Test EMEstimator with default parameters."""
        em_est = EMEstimator()
        
        assert em_est.max_iteration == 1000
        assert em_est.tolerance == 1e-4
        assert em_est.min_iteration == 1


class TestParameterUtils:
    """Test parameter utility functions."""
    
    def test_parameter_validation_valid(self):
        """Test parameter validation with valid inputs."""
        channel_pars = np.random.randn(3, 10)
        time_pars = np.random.rand(4, 2) + 0.1  # Positive values
        
        # Should not raise
        validate_parameters(
            channel_pars=channel_pars,
            time_pars=time_pars,
            n_events=3,
            n_dims=10,
            n_stages=4
        )

    def test_parameter_validation_invalid_shapes(self):
        """Test parameter validation with invalid shapes."""
        channel_pars = np.random.randn(2, 10)  # Wrong n_events
        time_pars = np.random.rand(4, 2) + 0.1
        
        with pytest.raises(ValueError, match="channel_pars shape"):
            validate_parameters(
                channel_pars=channel_pars,
                time_pars=time_pars,
                n_events=3,  # Expecting 3, got 2
                n_dims=10,
                n_stages=4
            )

    def test_parameter_validation_negative_time_pars(self):
        """Test parameter validation with negative time parameters."""
        channel_pars = np.random.randn(3, 10)
        time_pars = np.random.rand(4, 2)
        time_pars[0, 1] = -1.0  # Make scale parameter negative
        
        with pytest.raises(ValueError, match="Scale parameters"):
            validate_parameters(
                channel_pars=channel_pars,
                time_pars=time_pars,
                n_events=3,
                n_dims=10,
                n_stages=4
            )

    def test_parameter_initialization_random(self):
        """Test random parameter initialization."""
        channel_pars, time_pars = initialize_parameters(
            n_events=2, n_dims=5, method='random', random_seed=42
        )
        
        assert channel_pars.shape == (2, 5)
        assert time_pars.shape == (3, 2)  # n_events + 1
        assert np.all(time_pars > 0)  # All time parameters should be positive

    def test_parameter_initialization_uniform(self):
        """Test uniform parameter initialization."""
        channel_pars, time_pars = initialize_parameters(
            n_events=3, n_dims=4, method='uniform'
        )
        
        assert channel_pars.shape == (3, 4)
        assert time_pars.shape == (4, 2)
        assert np.allclose(channel_pars, 0.1)  # Should be all 0.1

    def test_log_likelihood_computation(self):
        """Test log-likelihood computation."""
        # Create fake event probabilities
        eventprobs = np.random.rand(50, 10, 3)
        eventprobs = eventprobs / eventprobs.sum(axis=0, keepdims=True)
        
        log_lkh = compute_log_likelihood(eventprobs)
        
        assert isinstance(log_lkh, float)
        assert not np.isnan(log_lkh)
        assert not np.isinf(log_lkh)


class TestConvergenceCheckers:
    """Test convergence checking utilities."""
    
    def test_relative_likelihood_convergence(self):
        """Test relative likelihood convergence checker."""
        checker = RelativeLikelihoodConvergence(tolerance=1e-4, min_iterations=3)
        
        # Simulate converged scenario
        likelihoods = [-1000.0, -999.9, -999.85, -999.8499]
        
        for i, lkh in enumerate(likelihoods[1:], 1):
            converged = checker.check_convergence(lkh, likelihoods[:i])
            if i < 3:
                assert not converged  # Should not converge before min_iterations
            elif i == 3:
                assert converged  # Should converge at iteration 3

    def test_relative_likelihood_no_early_convergence(self):
        """Test that convergence doesn't happen too early."""
        checker = RelativeLikelihoodConvergence(tolerance=1e-4, min_iterations=5)
        
        # Very similar likelihoods but below min_iterations
        likelihoods = [-1000.0, -999.99999]
        
        converged = checker.check_convergence(likelihoods[1], likelihoods[:1])
        assert not converged  # Should not converge before min_iterations


@pytest.mark.skipif(
    not PYMC_AVAILABLE,  # Skip if PyMC is not available
    reason="PyMC not available"
)
class TestMCMCEstimator:
    """Test MCMCEstimator class (requires PyMC)."""
    
    def test_mcmc_creation(self):
        """Test MCMCEstimator creation."""
        mcmc_est = MCMCEstimator(n_samples=100, n_chains=2)
        assert mcmc_est.n_samples == 100
        assert mcmc_est.n_chains == 2
        assert mcmc_est.supports_uncertainty() is True

    def test_mcmc_fit(self):
        """Test basic MCMC fit with synthetic data."""
        # Generate synthetic data
        n_events = 3
        eeg_data, _, _, event_width = demo(cpus=1, n_events=n_events, seed=42, overwrite=True)

        # Initialize model and estimator
        model = EventModel(n_events=n_events, sfreq=eeg_data.sfreq, event_width=event_width)
        mcmc_est = MCMCEstimator(n_samples=50, n_tune=50, n_chains=1, random_seed=42)

        # Initial parameters (can be random or from a quick EM fit)
        initial_channel_pars = np.random.randn(n_events, len(eeg_data.channel)) * 0.1
        initial_time_pars = np.array([[2, 10]] * (n_events + 1))

        # Preprocess the data
        preprocessed_data = Standard(eeg_data, n_comp=5, apply_zscore=False).data

        # Fit the model
        result = mcmc_est.fit(preprocessed_data, initial_channel_pars, initial_time_pars, model=model)

        # Assertions
        assert isinstance(result, EstimationResult)
        assert result.channel_pars.shape == (n_events, len(eeg_data.channel))
        assert result.time_pars.shape == (n_events + 1, 2)
        assert isinstance(result.likelihood, float)
        assert result.converged is False  # With few samples, convergence is unlikely
        assert "trace" in result.diagnostics
        assert "posterior_samples" in result.uncertainty

        # Test posterior samples shape
        assert result.uncertainty["posterior_samples"]["channel_pars"].shape == (1, 50, n_events, len(eeg_data.channel))
        assert result.uncertainty["posterior_samples"]["time_pars"].shape == (1, 50, n_events + 1, 2)

        # Test summary and plot_trace (just ensure they don't raise errors)
        mcmc_est.summary()
        mcmc_est.plot_trace()


if __name__ == "__main__":
    pytest.main([__file__])