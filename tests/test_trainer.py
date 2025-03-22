import torch
import pytest
import numpy as np
from nca.model import NCA
from nca.trainer import NCATrainer


def test_add_noise():
    """Test adding noise to state."""
    # Create a trainer with a simple model and target
    batch_size = 1
    channels = 3
    height, width = 10, 10
    
    # Create a model
    model = NCA(channels)
    
    # Create a target
    target = torch.zeros(batch_size, channels, height, width)
    
    # Create trainer with explicit CPU device for testing
    trainer = NCATrainer(model, target, device="cpu")
    
    # Create a state
    state = torch.zeros_like(target)
    
    # Test small noise
    sigma_small = 0.01
    noisy_state = trainer.add_noise(state, sigma_small)
    
    # Check that noise was added (state is no longer all zeros)
    assert not torch.allclose(noisy_state, state), "No noise was added"
    
    # Check that noise magnitude is approximately correct
    # For Gaussian noise, most values should be within 3*sigma of mean
    assert torch.mean(torch.abs(noisy_state - state)) < 3 * sigma_small
    
    # Test large noise
    sigma_large = 0.1
    noisy_state_large = trainer.add_noise(state, sigma_large)
    
    # Check that large noise has larger magnitude
    assert torch.mean(torch.abs(noisy_state_large)) > torch.mean(torch.abs(noisy_state))


def test_initialize_state():
    """Test initializing random state."""
    # Create a trainer with a simple model and target
    batch_size = 2
    channels = 3
    height, width = 8, 8
    
    # Create a model
    model = NCA(channels)
    
    # Create a target
    target = torch.zeros(batch_size, channels, height, width)
    
    # Create trainer with explicit CPU device for testing
    trainer = NCATrainer(model, target, device="cpu")
    
    # Initialize state
    state = trainer.initialize_state()
    
    # Check shape
    assert state.shape == target.shape, f"Expected shape {target.shape}, got {state.shape}"
    
    # Check that it's random (not all zeros or ones)
    assert not torch.allclose(state, torch.zeros_like(state))
    assert not torch.allclose(state, torch.ones_like(state))
    
    # Check that values are approximately standard normal
    assert -3 < torch.mean(state) < 3, "Mean is not close to 0"
    assert 0.5 < torch.std(state) < 1.5, "Standard deviation is not close to 1"


def test_train_step():
    """Test a single training step."""
    # Create a trainer with a simple model and target
    batch_size = 1
    channels = 3
    height, width = 5, 5
    
    # Create a model
    model = NCA(channels)
    
    # Create a target (all zeros)
    target = torch.zeros(batch_size, channels, height, width)
    
    # Create trainer with high epsilon to trigger reset more easily
    # Use CPU for testing
    epsilon = 0.1
    trainer = NCATrainer(model, target, epsilon=epsilon, device="cpu")
    
    # Create an initial state
    state = torch.ones_like(target)  # All ones (far from target)
    
    # Perform a training step
    next_state, loss, reset = trainer.train_step(state)
    
    # Check that loss was computed
    assert loss > 0, "Loss should be positive for non-matching state and target"
    
    # Check that next_state is different from state
    assert not torch.allclose(next_state, state), "State was not updated"
    
    # Check reset flag (should be False since loss is likely > epsilon)
    assert reset == (loss < epsilon), "Reset flag does not match loss < epsilon condition"


def test_reset_mechanism():
    """Test the reset mechanism when state gets close to target."""
    # Create a trainer with a simple model and target
    batch_size = 1
    channels = 3
    height, width = 5, 5
    
    # Create a model
    model = NCA(channels)
    
    # Create a target (all zeros)
    target = torch.zeros(batch_size, channels, height, width)
    
    # Create trainer with very high epsilon to force reset
    # Use CPU for testing
    epsilon = 100.0  # Any loss will be less than this
    sigma_large = 0.5  # Large noise for clear detection
    trainer = NCATrainer(
        model, target, 
        epsilon=epsilon, 
        sigma_large=sigma_large,
        device="cpu"
    )
    
    # Create a state close to target
    state = torch.zeros_like(target)  # Exactly the target
    
    # Save original state
    original_state = state.clone()
    
    # Perform a training step (should trigger reset)
    next_state, loss, reset = trainer.train_step(state)
    
    # Reset should be True since loss < epsilon
    assert reset, "Reset was not triggered even though loss < epsilon"
    
    # Next state should have large noise added
    assert not torch.allclose(next_state, original_state, atol=0.1), "Large noise not applied"
    
    # Difference should be approximately sigma_large in magnitude
    diff = next_state - original_state
    assert torch.mean(torch.abs(diff)) < 3 * sigma_large, "Noise magnitude incorrect" 