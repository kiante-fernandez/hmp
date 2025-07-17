"""Tests for the MCMCEstimator."""

import numpy as np
import pytest

from hmp.estimators import MCMCEstimator, EstimationResult
from hmp.models import EventModel
from hmp.trialdata import TrialData

# Check if PyMC is available
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

@pytest.fixture(scope="module")
def dummy_data_and_model():
    """Fixture to create dummy TrialData and an EventModel."""
    if not PYMC_AVAILABLE:
        pytest.skip("PyMC not available, skipping MCMC tests.")

    # Create dummy TrialData
    n_trials = 10
    n_channels = 3
    n_samples_per_trial = 100
    data = np.random.randn(n_trials, n_channels, n_samples_per_trial)

    # Generate dummy data for TrialData constructor
    sfreq = 100.0
    offset = 0
    starts = np.arange(n_trials) * n_samples_per_trial
    ends = starts + n_samples_per_trial - 1
    cross_corr = np.random.randn(n_trials * n_samples_per_trial, n_channels)

    # Create a dummy xr.DataArray for xrdurations
    import xarray as xr
    xrdurations = xr.DataArray(
        np.ones(n_trials) * n_samples_per_trial,
        coords={'trial': np.arange(n_trials)},
        dims=['trial']
    )

    trial_data = TrialData(
        xrdurations=xrdurations,
        starts=starts,
        ends=ends,
        n_trials=n_trials,
        n_samples=n_trials * n_samples_per_trial,
        sfreq=sfreq,
        offset=offset,
        cross_corr=cross_corr
    )

    # Create a simple EventModel
    n_events = 2
    n_stages = n_events + 1  # n_stages should be n_events + 1
    
    # Create a proper pattern object instead of just numpy array
    from hmp.patterns import HalfSine
    pattern = HalfSine.create_expected(sfreq=sfreq)
    
    model = EventModel(pattern, n_events=n_events)
    model.n_dims = n_channels # Manually set n_dims as it's usually set during model.fit

    # Initialize parameters for the model (these would typically come from a prior EM run or random init)
    initial_channel_pars = np.random.randn(n_events, n_channels) * 0.1  # Make smaller to avoid extreme values
    initial_time_pars = np.random.rand(n_stages, 2) * 2 + 1  # Ensure shape and scale are > 1

    return trial_data, model, initial_channel_pars, initial_time_pars

@pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not available, skipping MCMC tests.")
class TestMCMCEstimator:
    """Test MCMCEstimator class."""

    def test_mcmc_creation(self):
        """Test MCMCEstimator creation with parameters."""
        mcmc_est = MCMCEstimator(n_samples=100, n_chains=2, random_seed=42)
        assert mcmc_est.n_samples == 100
        assert mcmc_est.n_chains == 2
        assert mcmc_est.random_seed == 42
        assert mcmc_est.supports_uncertainty() is True

    def test_mcmc_fit(self, dummy_data_and_model):
        """Test fitting the MCMCEstimator."""
        trial_data, model, initial_channel_pars, initial_time_pars = dummy_data_and_model

        mcmc_est = MCMCEstimator(n_samples=50, n_tune=50, n_chains=2, random_seed=42) # Reduced samples for faster test

        result = mcmc_est.fit(
            trial_data=trial_data,
            initial_channel_pars=initial_channel_pars,
            initial_time_pars=initial_time_pars,
            model=model
        )

        assert isinstance(result, EstimationResult)
        assert result.channel_pars.shape == initial_channel_pars.shape
        assert result.time_pars.shape == initial_time_pars.shape
        assert isinstance(result.likelihood, float)
        assert isinstance(result.converged, bool)  # Check if it's a boolean
        assert result.n_iterations == mcmc_est.n_samples

        # Check uncertainty and diagnostics
        assert result.uncertainty is not None
        assert "uncertainty" not in result.diagnostics  # uncertainty is separate from diagnostics
        assert "rhat" in result.diagnostics
        assert "ess" in result.diagnostics
        assert "trace" in result.diagnostics

    def test_mcmc_fit_no_model_error(self, dummy_data_and_model):
        """Test that fit raises ValueError if no model is provided."""
        trial_data, _, initial_channel_pars, initial_time_pars = dummy_data_and_model
        mcmc_est = MCMCEstimator(random_seed=42)
        with pytest.raises(ValueError, match="MCMCEstimator requires a model instance"):
            mcmc_est.fit(
                trial_data=trial_data,
                initial_channel_pars=initial_channel_pars,
                initial_time_pars=initial_time_pars,
                model=None
            )

    def test_get_posterior_samples(self, dummy_data_and_model):
        """Test retrieving posterior samples."""
        trial_data, model, initial_channel_pars, initial_time_pars = dummy_data_and_model
        mcmc_est = MCMCEstimator(n_samples=50, n_tune=50, n_chains=2, random_seed=42)
        mcmc_est.fit(trial_data, initial_channel_pars, initial_time_pars, model)

        channel_samples = mcmc_est.get_posterior_samples(parameter="channel_pars")
        assert isinstance(channel_samples, np.ndarray)
        assert channel_samples.shape[2:] == initial_channel_pars.shape # (chains, draws, n_events, n_channels)

        all_samples = mcmc_est.get_posterior_samples()
        assert isinstance(all_samples, dict)
        assert "channel_pars" in all_samples
        assert "time_pars" in all_samples

        with pytest.raises(ValueError, match="Parameter 'non_existent_param' not found"):
            mcmc_est.get_posterior_samples(parameter="non_existent_param")

    def test_summary_and_plot_trace(self, dummy_data_and_model, capsys):
        """Test summary and plot_trace methods."""
        trial_data, model, initial_channel_pars, initial_time_pars = dummy_data_and_model
        mcmc_est = MCMCEstimator(n_samples=50, n_tune=50, n_chains=2, random_seed=42)
        mcmc_est.fit(trial_data, initial_channel_pars, initial_time_pars, model)

        # Test summary (capturing stdout)
        summary_str = mcmc_est.summary()
        assert isinstance(summary_str, str)
        assert "channel_pars" in summary_str
        assert "time_pars" in summary_str

        # Test plot_trace (just ensure it doesn't raise an error)
        # This will open a plot window, which is not ideal for automated tests.
        # We'll just call it and ensure no exceptions are raised.
        # In a real CI, you might want to mock matplotlib/arviz plotting functions.
        try:
            mcmc_est.plot_trace(parameter="channel_pars")
            mcmc_est.plot_trace() # Plot all
        except Exception as e:
            pytest.fail(f"plot_trace raised an exception: {e}")
