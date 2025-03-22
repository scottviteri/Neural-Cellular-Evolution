# Neural Cellular Automaton with Convolutional Attention: Design Doc

## Overview
This project implements a Neural Cellular Automaton (NCA) in PyTorch to evolve a random initial state into a target image through iterative, local updates. The model uses a **convolutional attention mechanism** for dynamic, non-linear transformations and trains sequentially on a **single long trajectory**. To simulate multiple trajectories, we inject a **small constant noise** at each step and apply a **large noise spike** whenever the state comes within a small epsilon of the target. Gradients are cut between steps to keep the optimization "shortsighted," focusing only on the current transition. This write-up provides a complete roadmap for development, including architecture, training, visualization, and testing.

### Objectives
- Transform a random initial state into a target image using an NCA.
- Use convolutional attention for expressive, local updates.
- Train on a single trajectory with noise-based simulated resets to mimic multiple trajectories.
- Keep gradients cut between steps for simplicity and computational efficiency.
- Provide clear visualization and robust testing to monitor progress and ensure correctness.

### Key Features
- **Convolutional Attention**: Each cell attends to its 3x3 neighborhood to compute updates.
- **Residual Updates**: The next state is the current state plus the attention output.
- **Noise Injection**: Small constant noise at each step, with large spikes near the target.
- **Shortsighted Training**: Gradients are cut between steps, optimizing only the current update.
- **Single Trajectory**: Simulates multiple trajectories via noise spikes within one sequence.

---

## Model Architecture
The NCA is a PyTorch module that defines a transition function `f` to update the state grid.

### Transition Function (`f`)
- **Input**: Current state `s_prev`, a tensor of shape `(batch_size, channels, height, width)`, where `channels` matches the target image (e.g., 3 for RGB).
- **Convolutional Attention Mechanism**:
  1. Extract the 3x3 neighborhood for each cell (e.g., using `unfold` or a custom convolution).
  2. Compute **query** (`q`), **key** (`k`), and **value** (`v`) vectors for the center cell and its neighbors via learned linear layers (e.g., 16-dimensional vectors).
  3. Calculate attention scores: `scores = q · k` (dot product between query and each key).
  4. Apply softmax to `scores` to get attention weights.
  5. Compute the update as the weighted sum of value vectors.
- **Output**: Update tensor, same shape as input.
- **Update Rule**: `s_next = s_prev + f(s_prev)` (residual connection).

### Implementation Notes
- Use a small hidden state (e.g., 16 channels) for `q`, `k`, and `v` to balance expressiveness and compute cost.
- Ensure locality by restricting attention to the 3x3 neighborhood.

---

## Training Procedure
Training occurs sequentially on a single long trajectory, with noise injection and simulated resets.

### Steps
1. **Setup**:
   - Define the target image as a tensor `(batch_size, channels, height, width)` (e.g., a 32x32 RGB image).
   - Initialize the NCA model with random weights.
   - Set an optimizer (e.g., Adam, learning rate 0.001).

2. **State Initialization**:
   - Start with a random state `s0` (e.g., standard Gaussian noise).

3. **Training Loop**:
   - **For** a fixed number of steps (e.g., 10,000):
     - Add small constant Gaussian noise: `s_prev = s_prev + noise`, where `noise ~ N(0, sigma_small)`.
     - Compute the next state: `s_next = s_prev + f(s_prev).detach()` (detach cuts gradients).
     - Calculate the L2 loss: `loss = MSE(s_next, target)`.
     - If `loss < epsilon` (e.g., 0.001):
       - Spike the noise: `s_prev = s_prev + noise_large`, where `noise_large ~ N(0, sigma_large)`.
     - Backpropagate through the current step and update the model parameters.
     - Update the state: `s_prev = s_next`.

4. **Stopping**:
   - Run for a predefined number of steps or until a maximum number of noise spikes (simulated trajectories) is reached.

### Noise Details
- **Small Constant Noise**: `sigma_small = 0.01` (keeps the state dynamic).
- **Large Noise Spike**: `sigma_large = 0.1` (simulates a new random starting point).
- **Epsilon**: `epsilon = 0.001` (threshold for considering the state "close" to the target).

### Why It Works
- The small noise encourages exploration and robustness.
- The large spike when `loss < epsilon` resets the state, simulating a new trajectory without restarting the loop.
- Cutting gradients keeps training focused on the current step, avoiding the complexity of backpropagating through the entire sequence.

---

## Visualization
Visualization helps track the NCA’s progress and debug issues.

### State Snapshots
- **Method**: Display or save the current state `s_prev` periodically.
- **Implementation**:
  - Use Matplotlib to plot the state as an image every 100 steps.
  - Ensure values are in [0,1] for proper rendering (e.g., sigmoid).
- **Purpose**: See how the state evolves toward the target.

### Loss Plot
- **Method**: Track the L2 loss over time.
- **Implementation**:
  - Log the loss at each step and plot it (e.g., with Matplotlib or TensorBoard).
- **Purpose**: Confirm the model is learning and detect when it reaches the target.

---

## Testing
Use pytests validate the implementation and ensure it behaves as expected.

### Unit Tests
- **Convolutional Attention**:
  - Input: A small 5x5 state tensor with known values.
  - Check: Output shape matches input, and attention weights are normalized.
- **Noise Injection**:
  - Apply small and large noise to a state, verify the variance matches `sigma_small` and `sigma_large`.

### Integration Test
- **Simple Target**:
  - Train on a uniform color image (e.g., all red).
  - Check: Loss drops below `epsilon` within a reasonable number of steps (e.g., 1000).
- **Noise Spike Behavior**:
  - Verify that when `loss < epsilon`, the state shifts significantly due to the large noise spike.

---

## Hyperparameters
Tune these based on experimentation:
- `sigma_small`: Start at 0.01; adjust if the state stagnates or overshoots.
- `sigma_large`: Start at 0.1; ensure it’s large enough to simulate a new trajectory.
- `epsilon`: Start at 0.001; tweak based on desired accuracy.
- Learning rate: Start at 0.001; adjust for convergence speed.

---

## Getting Started
- **Initial Setup**: Use a small grid (e.g., 32x32) and a simple target (e.g., a solid color).
- **First Run**: Train for 1000 steps, visualize every 100 steps, and check if the loss decreases.
- **Iterate**: Adjust noise levels or epsilon if the model struggles to converge or reset appropriately.

---

## Summary
This NCA uses convolutional attention to perform local, non-linear updates, trained sequentially with cut gradients for simplicity. The single-trajectory approach with constant small noise and large spikes near the target cleverly simulates multiple trajectories, making it both efficient and innovative. Visualization and testing ensure the process is transparent and reliable. 
