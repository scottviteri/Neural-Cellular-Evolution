import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess_target(image):
    """
    Preprocess target images for better gradient properties.
    Transforms values from [0,1] to an unbounded logit space.
    
    Args:
        image: Tensor with values in range [0,1]
        
    Returns:
        Tensor in an unbounded space suitable for neural network training
    """
    # Scale from [0,1] to [-0.9, 0.9]
    scaled = image * 1.8 - 0.9
    # Apply logit transformation, but ensure values are in a safe range first
    safe_scaled = scaled * 0.5 + 0.5  # Map to [0.05, 0.95] to avoid infinity issues
    return torch.logit(safe_scaled)


def postprocess_output(output):
    """
    Convert model output back to image space [0,1].
    Inverse of preprocess_target.
    
    Args:
        output: Tensor in unbounded space
        
    Returns:
        Tensor with values in range [0,1]
    """
    # Apply sigmoid to map back to [0,1]
    sigmoided = torch.sigmoid(output)
    # Rescale from [0,1] to original range
    return (sigmoided - 0.5) / 0.5 * 0.9 + 0.5


class ConvAttention(nn.Module):
    """
    Fast implementation of convolutional attention using unfold operations.
    This implementation properly handles boundaries and excludes padded elements from softmax.
    """
    def __init__(self, channels, hidden_dim=None):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim or channels  # Use channels if no hidden_dim
        
        # QKV projections
        self.query = nn.Linear(channels, self.hidden_dim)
        self.key = nn.Linear(channels, self.hidden_dim)
        self.value = nn.Linear(channels, self.hidden_dim)
        
        # Output projection to map back to original channel dimension if different
        self.output_proj = nn.Linear(self.hidden_dim, channels) if self.hidden_dim != channels else None
    
    def forward(self, x):
        """
        Fast implementation that properly handles boundaries and excludes padded elements from softmax.
        """
        B, C, H, W = x.shape
        
        # Step 1: Prepare query, key, value
        # Convert to shape [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Apply projections
        q_flat = self.query(x_flat)  # [B, H*W, hidden_dim]
        k_flat = self.key(x_flat)    # [B, H*W, hidden_dim]
        v_flat = self.value(x_flat)  # [B, H*W, hidden_dim]
        
        # Reshape for spatial operations
        q = q_flat.view(B, H, W, self.hidden_dim).permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]
        k = k_flat.view(B, H, W, self.hidden_dim).permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]
        v = v_flat.view(B, H, W, self.hidden_dim).permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]
        
        # Use unfold to extract 3x3 neighborhoods
        # Shape: [B, hidden_dim*9, H*W]
        k_neighborhoods = F.unfold(k, kernel_size=3, padding=1, stride=1)
        v_neighborhoods = F.unfold(v, kernel_size=3, padding=1, stride=1)
        
        # Reshape to [B, hidden_dim, 9, H*W]
        k_neighborhoods = k_neighborhoods.view(B, self.hidden_dim, 9, H*W)
        v_neighborhoods = v_neighborhoods.view(B, self.hidden_dim, 9, H*W)
        
        # Transpose to [B, H*W, 9, hidden_dim]
        k_neighborhoods = k_neighborhoods.permute(0, 3, 2, 1)
        v_neighborhoods = v_neighborhoods.permute(0, 3, 2, 1)
        
        # Reshape queries for batch matrix multiplication
        # [B, hidden_dim, H*W] -> [B, H*W, hidden_dim]
        q_reshaped = q.reshape(B, self.hidden_dim, H*W).permute(0, 2, 1)
        
        # Prepare q for batch matrix multiplication
        # [B, H*W, hidden_dim] -> [B, H*W, 1, hidden_dim]
        q_expanded = q_reshaped.unsqueeze(2)
        
        # Prepare k for batch matrix multiplication
        # [B, H*W, 9, hidden_dim] -> [B, H*W, hidden_dim, 9]
        k_transposed = k_neighborhoods.transpose(2, 3)
        
        # Compute attention scores
        # [B, H*W, 1, hidden_dim] @ [B, H*W, hidden_dim, 9] -> [B, H*W, 1, 9]
        scores = torch.matmul(q_expanded, k_transposed)
        scores = scores.squeeze(2) / (self.hidden_dim ** 0.5)  # [B, H*W, 9]
        
        # Create position-specific masks to exclude padded elements
        # First, convert linear position to 2D coordinates
        pos_y = torch.arange(H).view(1, H, 1).expand(1, H, W).reshape(-1).to(x.device)
        pos_x = torch.arange(W).view(1, 1, W).expand(1, H, W).reshape(-1).to(x.device)
        
        # For each position, determine which elements in the 3x3 grid are valid
        # The 3x3 grid is laid out as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        valid_mask = torch.zeros(H*W, 9, dtype=torch.bool, device=x.device)
        
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
        
        # Apply the mask to scores before softmax
        # Use a large negative value for invalid positions
        masked_scores = scores.clone()
        masked_scores = masked_scores.masked_fill(~valid_mask.unsqueeze(0).expand(B, -1, -1), -1e9)
        
        # Apply softmax to get attention weights
        # This will now only consider valid elements
        weights = F.softmax(masked_scores, dim=2).unsqueeze(2)  # [B, H*W, 1, 9]
        
        # Apply attention to values
        # [B, H*W, 1, 9] @ [B, H*W, 9, hidden_dim] -> [B, H*W, 1, hidden_dim]
        out = torch.matmul(weights, v_neighborhoods).squeeze(2)  # [B, H*W, hidden_dim]
        
        # Final projection if needed
        if self.output_proj is not None:
            out = self.output_proj(out)  # [B, H*W, C]
        
        # Reshape back to original format [B, C, H, W]
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C or hidden_dim, H, W]
        
        return out


class MLP(nn.Module):
    """
    Multi-layer perceptron with GELU activation.
    
    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (can be larger than in_features)
        out_features: Output dimension
        dropout: Dropout probability
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4  # Default to 4x expansion
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with convolutional attention and MLP.
    
    Args:
        channels: Number of channels
        hidden_dim: Hidden dimension for attention
        mlp_ratio: Ratio for MLP hidden dimension compared to input dimension
        dropout: Dropout probability
    """
    def __init__(self, channels, hidden_dim=None, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim or channels
        
        # Attention layer
        self.attention = ConvAttention(channels, hidden_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # MLP
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = MLP(
            in_features=channels,
            hidden_features=mlp_hidden_dim,
            out_features=channels,
            dropout=dropout
        )
        
    def forward(self, x):
        # Save original shape for residual connection
        B, C, H, W = x.shape
        
        # Attention block with residual connection
        # Reshape to [B, H*W, C] for LayerNorm
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Apply LayerNorm before attention (pre-norm formulation)
        norm_x = self.norm1(x_flat)
        
        # Reshape back to [B, C, H, W] for attention
        norm_x = norm_x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Apply attention and add residual
        x_flat = x_flat + self.attention(norm_x).permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Apply MLP with residual connection
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back to [B, C, H, W]
        x = x_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return x


class NCA(nn.Module):
    """
    Neural Cellular Automaton with convolutional attention and MLP.
    
    The model supports having more channels than just RGB (first 3 channels).
    Only the first 3 channels (RGB) are used for loss calculation and visualization,
    while additional channels serve as hidden state for information passing between steps.
    
    Args:
        channels: Number of channels in the state (can be more than 3)
        hidden_dim: Dimension of the hidden representation for attention (default: 16)
        num_blocks: Number of transformer blocks to use (default: 1)
        mlp_ratio: Ratio for MLP hidden dimension compared to input dimension (default: 4.0)
        dropout: Dropout probability (default: 0.0)
        use_nonlinearity: Whether to use a nonlinearity at the output (default: False)
        nonlin_type: Type of nonlinearity to use ('tanh', 'relu', 'leaky_relu') (default: 'tanh')
    """
    def __init__(self, channels, hidden_dim=16, num_blocks=1, mlp_ratio=4.0, dropout=0.0, 
                 use_nonlinearity=False, nonlin_type='tanh'):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_nonlinearity = use_nonlinearity
        
        # Create transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                channels=channels,
                hidden_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output nonlinearity
        if use_nonlinearity:
            if nonlin_type == 'tanh':
                self.nonlinearity = nn.Tanh()
            elif nonlin_type == 'relu':
                self.nonlinearity = nn.ReLU()
            elif nonlin_type == 'leaky_relu':
                self.nonlinearity = nn.LeakyReLU(0.2)
            else:
                raise ValueError(f"Unknown nonlinearity type: {nonlin_type}")
        else:
            self.nonlinearity = None
    
    def forward(self, state):
        """
        Computes one step of the NCA update.
        
        Args:
            state: Current state tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Updated state tensor of the same shape
        """
        # Pass through each transformer block sequentially
        output = state
        for block in self.blocks:
            output = block(output)
        
        # Apply nonlinearity if specified
        if self.use_nonlinearity and self.nonlinearity is not None:
            output = self.nonlinearity(output)
            
        return output