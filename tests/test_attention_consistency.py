import torch
import pytest
import sys
import os

# Add the parent directory to sys.path to import from nca
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nca.model import ConvAttention

class TestAttentionUtil:
    """Test class for verifying ConvAttention functionality."""
    
    def setup_environment(self, batch_size=2, channels=4, height=6, width=6, hidden_dim=8):
        """Set up the environment for testing."""
        # Create a sample input
        torch.manual_seed(42)  # For reproducibility
        sample_input = torch.randn(batch_size, channels, height, width)
        
        # Create an instance of ConvAttention
        attn = ConvAttention(channels, hidden_dim)
        
        return {
            'input': sample_input,
            'attn': attn,
            'batch_size': batch_size,
            'channels': channels,
            'height': height, 
            'width': width,
            'hidden_dim': hidden_dim
        }
    
    def test_valid_mask_creation(self):
        """Test that the valid_mask correctly identifies valid neighborhood elements."""
        setup = self.setup_environment()
        sample_input = setup['input']
        
        # Extract batch and shape information
        B, C, H, W = sample_input.shape
        hidden_dim = setup['hidden_dim']
        
        # Create 2D coordinates
        pos_y = torch.arange(H).view(1, H, 1).expand(1, H, W).reshape(-1)
        pos_x = torch.arange(W).view(1, 1, W).expand(1, H, W).reshape(-1)
        
        # Create mask manually
        valid_mask = torch.zeros(H*W, 9, dtype=torch.bool)
        
        # Top row
        valid_mask[:, 0] = (pos_y > 0) & (pos_x > 0)         # Top-left
        valid_mask[:, 1] = (pos_y > 0)                       # Top-center
        valid_mask[:, 2] = (pos_y > 0) & (pos_x < W-1)       # Top-right
        
        # Middle row
        valid_mask[:, 3] = (pos_x > 0)                       # Middle-left
        valid_mask[:, 4] = torch.ones_like(pos_x, dtype=torch.bool)  # Center (always valid)
        valid_mask[:, 5] = (pos_x < W-1)                     # Middle-right
        
        # Bottom row
        valid_mask[:, 6] = (pos_y < H-1) & (pos_x > 0)       # Bottom-left
        valid_mask[:, 7] = (pos_y < H-1)                     # Bottom-center
        valid_mask[:, 8] = (pos_y < H-1) & (pos_x < W-1)     # Bottom-right
        
        # Check mask for corners
        # Top-left corner (0,0)
        tl_idx = 0
        assert not valid_mask[tl_idx, 0], "Top-left element should be invalid for top-left corner"
        assert not valid_mask[tl_idx, 1], "Top-center element should be invalid for top-left corner"
        assert not valid_mask[tl_idx, 2], "Top-right element should be invalid for top-left corner"
        assert not valid_mask[tl_idx, 3], "Middle-left element should be invalid for top-left corner"
        assert valid_mask[tl_idx, 4], "Center element should be valid for top-left corner"
        assert valid_mask[tl_idx, 5], "Middle-right element should be valid for top-left corner"
        assert not valid_mask[tl_idx, 6], "Bottom-left element should be invalid for top-left corner"
        assert valid_mask[tl_idx, 7], "Bottom-center element should be valid for top-left corner"
        assert valid_mask[tl_idx, 8], "Bottom-right element should be valid for top-left corner"
        
        # Top-right corner (0,W-1)
        tr_idx = W - 1
        assert not valid_mask[tr_idx, 0], "Top-left element should be invalid for top-right corner"
        assert not valid_mask[tr_idx, 1], "Top-center element should be invalid for top-right corner"
        assert not valid_mask[tr_idx, 2], "Top-right element should be invalid for top-right corner"
        assert valid_mask[tr_idx, 3], "Middle-left element should be valid for top-right corner"
        assert valid_mask[tr_idx, 4], "Center element should be valid for top-right corner"
        assert not valid_mask[tr_idx, 5], "Middle-right element should be invalid for top-right corner"
        assert valid_mask[tr_idx, 6], "Bottom-left element should be valid for top-right corner"
        assert valid_mask[tr_idx, 7], "Bottom-center element should be valid for top-right corner"
        assert not valid_mask[tr_idx, 8], "Bottom-right element should be invalid for top-right corner"
        
        # Bottom-left corner (H-1,0)
        bl_idx = (H-1) * W
        assert not valid_mask[bl_idx, 0], "Top-left element should be invalid for bottom-left corner"
        assert valid_mask[bl_idx, 1], "Top-center element should be valid for bottom-left corner"
        assert valid_mask[bl_idx, 2], "Top-right element should be valid for bottom-left corner"
        assert not valid_mask[bl_idx, 3], "Middle-left element should be invalid for bottom-left corner"
        assert valid_mask[bl_idx, 4], "Center element should be valid for bottom-left corner"
        assert valid_mask[bl_idx, 5], "Middle-right element should be valid for bottom-left corner"
        assert not valid_mask[bl_idx, 6], "Bottom-left element should be invalid for bottom-left corner"
        assert not valid_mask[bl_idx, 7], "Bottom-center element should be invalid for bottom-left corner"
        assert not valid_mask[bl_idx, 8], "Bottom-right element should be invalid for bottom-left corner"
        
        # Bottom-right corner (H-1,W-1)
        br_idx = H*W - 1
        assert valid_mask[br_idx, 0], "Top-left element should be valid for bottom-right corner"
        assert valid_mask[br_idx, 1], "Top-center element should be valid for bottom-right corner"
        assert not valid_mask[br_idx, 2], "Top-right element should be invalid for bottom-right corner"
        assert valid_mask[br_idx, 3], "Middle-left element should be valid for bottom-right corner"
        assert valid_mask[br_idx, 4], "Center element should be valid for bottom-right corner"
        assert not valid_mask[br_idx, 5], "Middle-right element should be invalid for bottom-right corner"
        assert not valid_mask[br_idx, 6], "Bottom-left element should be invalid for bottom-right corner"
        assert not valid_mask[br_idx, 7], "Bottom-center element should be invalid for bottom-right corner"
        assert not valid_mask[br_idx, 8], "Bottom-right element should be invalid for bottom-right corner"
        
        # Center position (H//2, W//2)
        center_idx = (H//2) * W + (W//2)
        for i in range(9):
            assert valid_mask[center_idx, i], f"Element {i} should be valid for center position"
    
    def test_forward_pass(self):
        """Test that the forward pass works and produces the expected output shape."""
        setup = self.setup_environment()
        sample_input = setup['input']
        attn = setup['attn']
        
        # Run forward pass
        with torch.no_grad():
            output = attn(sample_input)
        
        # Check output shape
        assert output.shape == sample_input.shape, "Output shape should match input shape"

if __name__ == "__main__":
    # Run tests
    test_instance = TestAttentionUtil()
    print("Testing valid mask creation...")
    test_instance.test_valid_mask_creation()
    print("Valid mask creation test passed!")
    
    print("\nTesting forward pass...")
    test_instance.test_forward_pass()
    print("Forward pass test passed!")
    
    print("\nAll tests passed!") 