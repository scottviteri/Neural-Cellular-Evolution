# Neural Cellular Evolution with Convolutional Attention

This repository implements Neural Cellular Automaton (NCA) with convolutional attention in PyTorch. The model uses a specialized attention mechanism to evolve random initial states into target images through iterative local updates, where each cell interacts with its 3x3 neighborhood to create complex emergent patterns.

## Core Features

- **Convolutional Attention Mechanism**: Efficiently implemented attention between local 3x3 neighborhoods with proper boundary handling
- **Transformer-style Architecture**: Configurable blocks with attention, optional nonlinearities, and MLP layers
- **Residual Updates**: States evolve gradually through residual connections
- **Noise Injection**: Small constant noise and large spike noise for exploration and simulated resets
- **Lambda Scheduling**: Various schedule types to control mixing between random initialization and previous states
- **Rich Visualizations**: Training animations, state snapshots, and diagnostic plots

## Installation

To install the package and its dependencies:

```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[dev]"
```

## Running the Model

### Quick Demo

For a quick demonstration:

```bash
python -m nca.main --demo
```

This will run the NCA training with optimized parameters (red circle target, 10000 steps, batch size 8) and save outputs to the default output directory.

### Custom Training

Customize training with various command line arguments:

```bash
# Basic training with a circle target
python -m nca.main --target-type circle --steps 10000 --batch-size 8 --output-dir my_results

# Using a custom image as target
python -m nca.main --target-image path/to/image.jpg --size 64 --channels 24

# Adjusting model architecture
python -m nca.main --hidden-dim 32 --num-blocks 2 --use-nonlinearity --nonlin-type relu
```

## Available Parameters

### Model Parameters
- `--hidden-dim`: Hidden dimension for attention mechanism (default: 16)
- `--num-blocks`: Number of transformer blocks (default: 1)
- `--mlp-ratio`: Ratio for MLP hidden dimension (default: 4.0)
- `--dropout`: Dropout probability (default: 0.0)
- `--use-nonlinearity`: Enable nonlinearity after attention layer
- `--nonlin-type`: Type of nonlinearity ["tanh", "relu", "leaky_relu"] (default: "tanh")

### Target Parameters
- `--target-image`: Path to target image
- `--target-type`: Simple target type ["solid", "gradient", "checkerboard", "circle"] (default: "circle")
- `--size`: Size of target image (default: 32)
- `--batch-size`: Batch size (default: 8)
- `--channels`: Total channels for model (default: 16)
- `--rgb-channels`: RGB channels for target (default: 3)

### Training Parameters
- `--steps`: Number of training steps (default: 10000)
- `--learning-rate`: Learning rate (default: 0.001)
- `--noise-small`: Small constant noise (default: 0.01)
- `--noise-large`: Large noise spikes (default: 0.1)
- `--epsilon`: Target proximity threshold (default: 0.001)
- `--init-type`: State initialization ["random", "target", "zeros", "ones"] (default: "random")
- `--filter-improvements`: Only update when MSE improves

### Lambda Schedule Parameters
- `--lambda-schedule`: Schedule type ["constant", "independent", "linear", "cosine", "exponential", "step"] (default: "constant")
- `--lambda-value`: Value for constant schedule (default: 1.0)
- `--lambda-start`: Starting value for non-constant schedules (default: 1.0)
- `--lambda-end`: Ending value for non-constant schedules (default: 0.0)
- `--warmup-steps`: Warmup steps for step schedule (default: steps/2)
- `--random-init-sigma`: Sigma for random initialization (default: 0.1)

### Output Parameters
- `--log-interval`: Console logging interval (default: 50)
- `--image-interval`: Image saving interval (default: 100)
- `--disable-gif`: Disable GIF creation
- `--output-dir`: Directory to save results (default: "output")

## Project Structure

- `nca/`: Main package
  - `model.py`: NCA model definition with transformer blocks and attention mechanism
  - `trainer.py`: Training loop with noise injection and lambda scheduling
  - `visualization.py`: Visualization utilities for images and animations
  - `main.py`: Command-line interface and configuration
- `tests/`: Unit tests for model components
  - `test_attention.py`: Tests for the attention mechanism
  - `test_attention_consistency.py`: Tests for attention behavior consistency
  - `test_conv_attention.py`: Tests for convolutional attention
  - `test_model.py`: Tests for the NCA model
  - `test_trainer.py`: Tests for the training procedure

## Testing

Run the test suite with:

```bash
pytest
```

## How It Works

The NCA uses convolutional attention to perform local, non-linear updates. Training happens on a single trajectory with noise injection to simulate multiple trajectories:

1. Small constant noise is added at each step to encourage exploration
2. When the state gets close to the target, a large noise spike simulates a new trajectory
3. Gradients are cut between steps to keep training focused on the current transition

<<<<<<< HEAD
The `attention_guided_nca.egg-info` directory is generated by the Python packaging system and can be safely ignored. 
=======
This approach makes training both efficient and innovative while producing complex emergent behaviors.

For more details, see the [design documentation](design_doc.md).
>>>>>>> da7d863 (update readme to include lambda schedule)
