"""Simplified JAX-based operations for HMP likelihood computation."""

import numpy as np
from typing import Tuple, Optional

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False


def check_jax_available():
    """Runtime check for JAX availability."""
    return JAX_AVAILABLE


if JAX_AVAILABLE:
    def _gamma_pdf_jax(x: jnp.ndarray, shape: float, scale: float) -> jnp.ndarray:
        """JAX implementation of Gamma PDF."""
        x = jnp.maximum(x, 1e-15)
        log_pdf = ((shape - 1) * jnp.log(x) - x / scale - 
                   shape * jnp.log(scale) - jax.scipy.special.gammaln(shape))
        return jnp.exp(log_pdf)

    def _distribution_pdf_jax(shape: float, scale: float, max_duration: int) -> jnp.ndarray:
        """JAX version of distribution PDF computation."""
        x = jnp.arange(1, max_duration + 1, dtype=jnp.float64)
        pdf = _gamma_pdf_jax(x, shape, scale)
        return pdf / jnp.sum(pdf)  # Normalize

    @jit
    def estim_probs_jax_simple(
        cross_corr: jnp.ndarray,
        channel_pars: jnp.ndarray,
        time_pars: jnp.ndarray,
        durations: jnp.ndarray,
        starts: jnp.ndarray,
        ends: jnp.ndarray,
        locations: jnp.ndarray
    ) -> float:
        """
        Simplified JAX version that just computes likelihood.
        Returns only the log-likelihood for gradient computation.
        """
        n_events, n_dims = channel_pars.shape
        n_trials = len(durations)
        n_stages = n_events + 1
        max_duration = jnp.max(durations).astype(int)
        n_samples = cross_corr.shape[0]

        # Compute gains
        gains = jnp.zeros((n_samples, n_events), dtype=jnp.float64)
        for i in range(n_dims):
            channel_contribution = (
                cross_corr[:, i][:, None] * channel_pars[:, i][None, :] -
                channel_pars[:, i][None, :] ** 2 / 2
            )
            gains = gains + channel_contribution
        gains = jnp.exp(gains)

        # Simplified likelihood computation for gradient purposes
        # Focus on the parts most important for parameter estimation
        
        # Simplified channel contribution to likelihood
        # Use vectorized computation instead of loops for JAX compatibility
        def compute_trial_contribution(trial_idx):
            start_idx = starts[trial_idx].astype(int)
            end_idx = ends[trial_idx].astype(int)
            
            # Ensure indices are within bounds
            start_idx = jnp.maximum(0, jnp.minimum(start_idx, n_samples - 1))
            end_idx = jnp.maximum(start_idx, jnp.minimum(end_idx, n_samples - 1))
            
            # Create mask for valid indices in this trial
            indices = jnp.arange(n_samples)
            mask = (indices >= start_idx) & (indices <= end_idx)
            
            # Sum gains for this trial using mask
            trial_contribution = jnp.sum(gains * mask[:, None])
            return trial_contribution
        
        # Use vmap to vectorize over trials
        trial_contributions = jax.vmap(compute_trial_contribution)(jnp.arange(n_trials))
        channel_likelihood = jnp.sum(trial_contributions)
        
        # Time parameter contribution (simplified gamma likelihood)
        def compute_stage_contribution(stage):
            shape_param = time_pars[stage, 0]
            scale_param = time_pars[stage, 1]
            
            # Simplified time contribution - use mean duration for each stage
            mean_duration = jnp.mean(durations) / n_stages
            
            # Gamma log-likelihood terms
            log_contrib = ((shape_param - 1) * jnp.log(jnp.maximum(mean_duration, 1e-6)) - 
                          mean_duration / jnp.maximum(scale_param, 1e-6))
            return log_contrib
        
        # Use vmap to vectorize over stages
        stage_contributions = jax.vmap(compute_stage_contribution)(jnp.arange(n_stages))
        time_likelihood = jnp.sum(stage_contributions)
        
        return channel_likelihood + time_likelihood

    # Create gradient function
    estim_probs_grad_jax = jit(value_and_grad(estim_probs_jax_simple, argnums=(1, 2)))

    def compute_hmp_likelihood_and_gradients(cross_corr, channel_pars, time_pars, 
                                           durations, starts, ends, locations):
        """Compute likelihood and gradients using JAX."""
        # Convert to JAX arrays
        cross_corr_jax = jnp.array(cross_corr, dtype=jnp.float64)
        channel_pars_jax = jnp.array(channel_pars, dtype=jnp.float64)
        time_pars_jax = jnp.array(time_pars, dtype=jnp.float64)
        durations_jax = jnp.array(durations, dtype=jnp.int32)
        starts_jax = jnp.array(starts, dtype=jnp.int32)
        ends_jax = jnp.array(ends, dtype=jnp.int32)
        locations_jax = jnp.array(locations, dtype=jnp.int32)
        
        # Compute likelihood and gradients
        likelihood, (channel_grad, time_grad) = estim_probs_grad_jax(
            cross_corr_jax, channel_pars_jax, time_pars_jax,
            durations_jax, starts_jax, ends_jax, locations_jax
        )
        
        return float(likelihood), np.array(channel_grad), np.array(time_grad)

else:
    # Fallback functions when JAX is not available
    def compute_hmp_likelihood_and_gradients(*args, **kwargs):
        raise ImportError("JAX is required for gradient computation but is not available")
    
    estim_probs_jax_simple = None
    estim_probs_grad_jax = None


# PyTensor integration
def create_jax_likelihood_op():
    """Create a PyTensor Op that uses JAX for likelihood computation."""
    try:
        import pytensor.tensor as at
        from pytensor.graph.op import Op
        from pytensor.graph.basic import Apply
        
        class JAXHMPLikelihoodOp(Op):
            """PyTensor Op that uses JAX for HMP likelihood computation."""
            
            def connection_pattern(self, node):
                """Specify how inputs connect to outputs for gradient computation."""
                # Only channel_pars (input 0) and time_pars (input 1) affect the output
                # Static inputs (cross_corr, durations, starts, ends, locations) don't
                return [
                    [True],   # channel_pars -> output
                    [True],   # time_pars -> output  
                    [False],  # cross_corr (static)
                    [False],  # durations (static)
                    [False],  # starts (static)
                    [False],  # ends (static)
                    [False]   # locations (static)
                ]
            
            def make_node(self, channel_pars, time_pars, cross_corr, durations, 
                         starts, ends, locations):
                # Convert inputs to tensor variables
                inputs = [
                    at.as_tensor_variable(channel_pars),
                    at.as_tensor_variable(time_pars),
                    at.as_tensor_variable(cross_corr),
                    at.as_tensor_variable(durations),
                    at.as_tensor_variable(starts),
                    at.as_tensor_variable(ends),
                    at.as_tensor_variable(locations)
                ]
                
                # Output is a scalar
                outputs = [at.dscalar()]
                
                return Apply(self, inputs, outputs)
            
            def perform(self, node, inputs, outputs):
                channel_pars, time_pars, cross_corr, durations, starts, ends, locations = inputs
                
                # Compute likelihood using JAX
                likelihood, _, _ = compute_hmp_likelihood_and_gradients(
                    cross_corr, channel_pars, time_pars, 
                    durations, starts, ends, locations
                )
                
                outputs[0][0] = np.asarray(likelihood, dtype=node.outputs[0].dtype)
            
            def grad(self, inputs, output_grads):
                import pytensor.tensor as at
                from pytensor.gradient import disconnected_type
                
                channel_pars, time_pars = inputs[0], inputs[1]
                static_inputs = inputs[2:]  # cross_corr, durations, etc.
                
                # Create gradient op
                grad_op = JAXHMPGradientOp()
                channel_grad, time_grad = grad_op(*inputs)
                
                output_grad = output_grads[0]
                
                gradients = [
                    output_grad * channel_grad,
                    output_grad * time_grad,
                    # Use disconnected_type() for static inputs as specified in connection_pattern
                    disconnected_type(),  # cross_corr
                    disconnected_type(),  # durations  
                    disconnected_type(),  # starts
                    disconnected_type(),  # ends
                    disconnected_type()   # locations
                ]
                
                return gradients
        
        class JAXHMPGradientOp(Op):
            """PyTensor Op for computing gradients using JAX."""
            
            def connection_pattern(self, node):
                """Specify how inputs connect to outputs for gradient computation."""
                # Only channel_pars (input 0) and time_pars (input 1) affect the outputs
                # Static inputs don't affect gradient outputs
                return [
                    [True, False],   # channel_pars -> channel_grad, not time_grad
                    [False, True],   # time_pars -> time_grad, not channel_grad
                    [False, False],  # cross_corr (static)
                    [False, False],  # durations (static)
                    [False, False],  # starts (static)
                    [False, False],  # ends (static)
                    [False, False]   # locations (static)
                ]
            
            def make_node(self, channel_pars, time_pars, cross_corr, durations, 
                         starts, ends, locations):
                inputs = [
                    at.as_tensor_variable(channel_pars),
                    at.as_tensor_variable(time_pars),
                    at.as_tensor_variable(cross_corr),
                    at.as_tensor_variable(durations),
                    at.as_tensor_variable(starts),
                    at.as_tensor_variable(ends),
                    at.as_tensor_variable(locations)
                ]
                
                # Outputs are gradients with same shape as parameters
                outputs = [channel_pars.type(), time_pars.type()]
                
                return Apply(self, inputs, outputs)
            
            def perform(self, node, inputs, outputs):
                channel_pars, time_pars, cross_corr, durations, starts, ends, locations = inputs
                
                # Compute gradients using JAX
                _, channel_grad, time_grad = compute_hmp_likelihood_and_gradients(
                    cross_corr, channel_pars, time_pars, 
                    durations, starts, ends, locations
                )
                
                outputs[0][0] = np.asarray(channel_grad, dtype=channel_pars.dtype)
                outputs[1][0] = np.asarray(time_grad, dtype=time_pars.dtype)
        
        return JAXHMPLikelihoodOp, JAXHMPGradientOp
        
    except ImportError:
        return None, None


# Global variables for Op classes
JAXHMPLikelihoodOp, JAXHMPGradientOp = create_jax_likelihood_op()