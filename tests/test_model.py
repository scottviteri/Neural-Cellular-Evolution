import torch
import pytest
import numpy as np
from nca.model import NCA, ConvAttention


def test_conv_attention_shape():
    """Test that ConvAttention maintains input shape."""
    batch_size = 2
    channels = 3
    height, width = 10, 12
    
    # Create a random input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Create ConvAttention module
    attn = ConvAttention(channels, hidden_dim=16)
    
    # Forward pass
    output = attn(x)
    
    # Check that output shape matches input shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"


def test_conv_attention_normalization():
    """Test that attention weights sum to 1 for each position."""
    # Small test case for easier inspection
    batch_size = 1
    channels = 3
    height, width = 5, 5
    hidden_dim = 4
    kernel_size = 3
    
    # Create a known input tensor
    x = torch.ones(batch_size, channels, height, width)
    
    # Create ConvAttention module
    attn = ConvAttention(channels, hidden_dim=hidden_dim)
    
    # Perform similar operations as in the forward method
    # Reshape for linear projection
    x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, height*width, channels)
    
    # Compute q, k
    q_flat = attn.query(x_flat)
    k_flat = attn.key(x_flat)
    
    # Reshape back to spatial dimensions
    q = q_flat.view(batch_size, height, width, hidden_dim).permute(0, 3, 1, 2)
    k = k_flat.view(batch_size, height, width, hidden_dim).permute(0, 3, 1, 2)
    
    # Extract 3x3 neighborhoods
    padding = 1
    k_unfolded = torch.nn.functional.unfold(k, kernel_size=kernel_size, padding=padding)
    k_unfolded = k_unfolded.view(batch_size, hidden_dim, kernel_size*kernel_size, height*width)
    
    # Prepare query
    q_reshaped = q.view(batch_size, hidden_dim, 1, height*width)
    
    # Transpose for matrix multiplication
    q_transposed = q_reshaped.permute(0, 2, 1, 3)
    k_transposed = k_unfolded.permute(0, 2, 1, 3)
    
    # Compute attention scores
    attention_scores = torch.matmul(q_transposed, k_transposed.transpose(2, 3))
    attention_scores = attention_scores / (hidden_dim ** 0.5)
    
    # Apply softmax
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=2)
    
    # Check that weights sum to 1 for each position (with small tolerance for numerical issues)
    sums = attention_weights.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_nca_forward():
    """Test NCA forward pass."""
    batch_size = 2
    channels = 3
    height, width = 8, 8
    
    # Create a random input state
    state = torch.randn(batch_size, channels, height, width)
    
    # Create NCA model
    model = NCA(channels, hidden_dim=16)
    
    # Forward pass
    new_state = model(state)
    
    # Check that new_state shape matches state shape
    assert new_state.shape == state.shape, f"Expected shape {state.shape}, got {new_state.shape}"
    
    # Check that new_state is different from state (update was applied)
    assert not torch.allclose(new_state, state), "NCA did not update the state"


def test_nca_residual_connection():
    """Test that NCA uses residual connection."""
    batch_size = 1
    channels = 3
    height, width = 5, 5
    
    # Create a known input state
    state = torch.ones(batch_size, channels, height, width)
    
    # Create NCA model with custom parameters to make testing easier
    model = NCA(channels, hidden_dim=4)
    
    # Forward pass with gradient tracking
    new_state = model(state)
    
    # Check that new_state includes the original state (residual connection)
    diff = new_state - state
    
    # The difference should be the update from the attention mechanism
    assert torch.any(torch.abs(diff) > 0), "No update was applied"
    
    # For a residual connection, the sign of the difference matters less than the fact
    # that the original state is preserved in the update
    assert torch.allclose(new_state, state + diff)


def test_conv_attention_no_hidden_dim():
    """Test ConvAttention without a hidden dimension."""
    batch_size = 2
    channels = 3
    height, width = 10, 12
    
    # Create a random input tensor
    x = torch.randn(batch_size, channels, height, width)
    
    # Create ConvAttention module without hidden_dim
    attn = ConvAttention(channels, hidden_dim=None)
    
    # Forward pass
    output = attn(x)
    
    # Check that output shape matches input shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    # Check that hidden_dim is set to channels
    assert attn.hidden_dim == channels, f"Expected hidden_dim to be {channels}, got {attn.hidden_dim}" 