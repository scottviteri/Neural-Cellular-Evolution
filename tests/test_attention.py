import torch
import pytest
import sys
import os

# Add the parent directory to sys.path to import from nca
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nca.model import ConvAttention

class TestConvAttention:
    @pytest.fixture
    def sample_input(self):
        # Create a small test input: batch_size=2, channels=4, height=6, width=6
        return torch.randn(2, 4, 6, 6)
    
    def test_qkv_projections(self):
        """Test that QKV projections produce correct shapes"""
        # Setup
        batch_size, channels, height, width = 2, 4, 6, 6
        hidden_dim = 8
        x = torch.randn(batch_size, channels, height, width)
        
        # With hidden dimension
        attn = ConvAttention(channels, hidden_dim)
        x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, height*width, channels)
        
        q = attn.query(x_flat)
        k = attn.key(x_flat)
        v = attn.value(x_flat)
        
        # Check shapes
        assert q.shape == (batch_size, height*width, hidden_dim)
        assert k.shape == (batch_size, height*width, hidden_dim)
        assert v.shape == (batch_size, height*width, hidden_dim)
        
        # Without hidden dimension
        attn = ConvAttention(channels)
        q = attn.query(x_flat)
        k = attn.key(x_flat)
        v = attn.value(x_flat)
        
        # Check shapes
        assert q.shape == (batch_size, height*width, channels)
        assert k.shape == (batch_size, height*width, channels)
        assert v.shape == (batch_size, height*width, channels)
    
    def test_unfold_operation(self):
        """Test the unfold operation for extracting 3x3 neighborhoods"""
        batch_size, channels, height, width = 2, 4, 6, 6
        x = torch.randn(batch_size, channels, height, width)
        
        # Apply unfold to extract 3x3 neighborhoods
        # kernel_size=3, padding=1, stride=1
        neighborhoods = torch.nn.functional.unfold(
            x, kernel_size=3, padding=1, stride=1
        )
        
        # Expected shape: [batch_size, channels*kernel_size*kernel_size, height*width]
        # = [2, 4*3*3, 6*6]
        expected_shape = (batch_size, channels*3*3, height*width)
        assert neighborhoods.shape == expected_shape
        
        # Verify we can reshape back to spatial dimensions
        neighborhoods = neighborhoods.reshape(batch_size, channels, 3*3, height*width)
        assert neighborhoods.shape == (batch_size, channels, 9, height*width)
    
    def test_attention_calculation(self):
        """Test attention score calculation and softmax application"""
        batch_size, hidden_dim = 2, 8
        height, width = 6, 6
        patch_size = 3 * 3  # 3x3 neighborhood
        
        # Create sample query and key
        q = torch.randn(batch_size, height*width, hidden_dim)
        # For each position, we have 9 keys from the 3x3 neighborhood
        k = torch.randn(batch_size, height*width, patch_size, hidden_dim)
        
        # Reshape for batch matrix multiplication
        q_expanded = q.unsqueeze(2)  # [B, H*W, 1, hidden_dim]
        k_transposed = k.transpose(2, 3)  # [B, H*W, hidden_dim, patch_size]
        
        # Calculate attention scores
        scores = torch.matmul(q_expanded, k_transposed)  # [B, H*W, 1, patch_size]
        scores = scores.squeeze(2) / (hidden_dim ** 0.5)  # [B, H*W, patch_size]
        
        # Apply softmax
        weights = torch.nn.functional.softmax(scores, dim=-1)  # [B, H*W, patch_size]
        
        assert weights.shape == (batch_size, height*width, patch_size)
        # Check softmax properties
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, height*width))
    
    def test_attention_application(self):
        """Test applying attention weights to values"""
        batch_size, hidden_dim = 2, 8
        height, width = 6, 6
        patch_size = 3 * 3  # 3x3 neighborhood
        
        # Create sample weights and values
        weights = torch.softmax(torch.randn(batch_size, height*width, patch_size), dim=-1)
        v = torch.randn(batch_size, height*width, patch_size, hidden_dim)
        
        # Apply attention to values
        context = torch.matmul(weights.unsqueeze(2), v)  # [B, H*W, 1, hidden_dim]
        context = context.squeeze(2)  # [B, H*W, hidden_dim]
        
        assert context.shape == (batch_size, height*width, hidden_dim)
    
    def test_output_shape(self):
        """Test that the ConvAttention produces outputs with the correct shape"""
        # Create a sample input with a fixed seed for reproducibility
        torch.manual_seed(42)
        batch_size, channels, height, width = 2, 4, 6, 6
        sample_input = torch.randn(batch_size, channels, height, width)
        
        # Create ConvAttention instance
        hidden_dim = 8
        attn = ConvAttention(channels, hidden_dim)
        
        # Compute output
        with torch.no_grad():
            output = attn(sample_input)
        
        # Check output shape
        assert output.shape == sample_input.shape, "Output shape should match input shape"
    
    def test_performance(self):
        """Measure performance of the ConvAttention implementation"""
        # Create a larger sample input for more meaningful timing
        batch_size, channels, height, width = 2, 16, 32, 32
        sample_input = torch.randn(batch_size, channels, height, width)
        
        hidden_dim = 16
        attn = ConvAttention(channels, hidden_dim)
        
        # Measure execution time
        import time
        
        # Warmup
        for _ in range(2):
            _ = attn(sample_input)
        
        # Measure execution time
        start_time = time.time()
        for _ in range(5):
            _ = attn(sample_input)
        execution_time = time.time() - start_time
        
        print("\nPerformance measurement:")
        print(f"ConvAttention execution time (5 runs): {execution_time:.4f}s")
        print(f"Average per run: {execution_time/5:.4f}s")

if __name__ == "__main__":
    # Run tests manually
    test_instance = TestConvAttention()
    test_instance.test_qkv_projections()
    test_instance.test_unfold_operation()
    test_instance.test_attention_calculation()
    test_instance.test_attention_application()
    test_instance.test_output_shape()
    test_instance.test_performance()
    print("\nAll tests passed!") 