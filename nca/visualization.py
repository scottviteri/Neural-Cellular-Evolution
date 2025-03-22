import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation


def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a numpy image array.
    
    Args:
        tensor: PyTorch tensor of shape (batch, channels, height, width)
        
    Returns:
        Numpy array of shape (height, width, channels)
    """
    # Take the first batch element
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU and convert to numpy
    img = tensor.detach().cpu().numpy()
    
    # Transpose from (channels, height, width) to (height, width, channels)
    img = np.transpose(img, (1, 2, 0))
    
    # Clip to [0, 1] range
    img = np.clip(img, 0, 1)
    
    return img


def visualize_state(state, target=None, ax=None, title=None):
    """
    Visualize the current state of the NCA.
    
    Args:
        state: PyTorch tensor of shape (batch, channels, height, width)
        target: Optional target tensor of the same shape
        ax: Optional matplotlib axis to plot on
        title: Optional title for the plot
        
    Returns:
        Matplotlib axis or tuple (axis, figure) depending on whether a new figure was created
    """
    created_fig = False
    if ax is None:
        created_fig = True
        if target is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(tensor_to_image(state))
            ax1.set_title("Current State")
            ax1.axis("off")
            
            ax2.imshow(tensor_to_image(target))
            ax2.set_title("Target")
            ax2.axis("off")
            
            if title:
                plt.suptitle(title)
            
            if created_fig:
                return (ax1, ax2), fig
            return ax1, ax2
        else:
            fig, ax = plt.subplots(figsize=(5, 5))
    
    # Convert tensor to image and display
    img = tensor_to_image(state)
    ax.imshow(img)
    
    if title:
        ax.set_title(title)
    
    ax.axis("off")
    
    if created_fig:
        return ax, fig
    return ax


def plot_loss(losses, reset_points=None, figsize=(10, 5)):
    """
    Plot the loss history during training.
    
    Args:
        losses: List of loss values
        reset_points: Optional list of indices where noise spikes were applied
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot loss curve
    ax.plot(losses, label="Loss")
    
    # Mark reset points if provided
    if reset_points is not None:
        for reset in reset_points:
            ax.axvline(x=reset, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    
    return fig


def create_animation(states, steps=None, losses=None, figsize=(7, 6), interval=100):
    """
    Create an animation from a list of state tensors.
    
    Args:
        states: List of state tensors
        steps: Optional list of step numbers corresponding to each state
        losses: Optional list of loss values corresponding to each state
        figsize: Figure size
        interval: Time between frames in milliseconds
        
    Returns:
        Matplotlib animation
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initialize with the first state
    img = tensor_to_image(states[0])
    im = ax.imshow(img)
    ax.axis("off")
    
    # Add a progress bar and text for step/loss information
    if steps is not None:
        # Create a progress bar at the bottom
        progress_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
        progress_bar = progress_ax.barh([0], [0], color='blue', height=1.0)
        
        # Create text for step and loss information
        if losses is not None:
            step_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, 
                               color='white', fontsize=12, 
                               bbox=dict(facecolor='black', alpha=0.7))
        else:
            step_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, 
                               color='white', fontsize=12, 
                               bbox=dict(facecolor='black', alpha=0.7))
    
    def update(frame):
        img = tensor_to_image(states[frame])
        im.set_array(img)
        
        # Update progress bar and text if steps are provided
        if steps is not None:
            current_step = steps[frame]
            max_step = steps[-1]
            progress_value = current_step / max_step if max_step > 0 else 0
            
            # Update progress bar
            progress_bar[0].set_width(progress_value)
            progress_ax.set_xlim(0, 1)
            progress_ax.set_xticks([])
            progress_ax.set_yticks([])
            
            # Update text
            if losses is not None:
                step_text.set_text(f"Step: {current_step}/{max_step}\nLoss: {losses[frame]:.6f}")
            else:
                step_text.set_text(f"Step: {current_step}/{max_step}")
            
            return [im, progress_bar[0], step_text]
        else:
            return [im]
    
    ani = FuncAnimation(
        fig, update, frames=range(len(states)), 
        blit=True, interval=interval
    )
    
    return ani


def create_training_callback(save_dir, save_interval=1000, gif_interval=None):
    """
    Create a callback function for visualization during training.
    
    Args:
        save_dir: Directory to save visualizations
        save_interval: Interval (in steps) to save state images for monitoring
        gif_interval: Interval for saving frames for GIF animation (if None, will use save_interval, if 0 will disable)
        
    Returns:
        Callback function for the trainer
    """
    import os
    from pathlib import Path
    
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Store the original gif_interval for later reference
    original_gif_interval = gif_interval
    
    # If gif_interval is 0, disable GIF frame collection
    # If gif_interval is None, use 10x save_interval
    if gif_interval == 0:
        print("GIF frame collection disabled")
        gif_interval = float('inf')  # Set to infinity to disable
    elif gif_interval is None:
        # Default: GIF interval is 10x the save interval
        gif_interval = save_interval * 10
        print(f"Using 10x save_interval ({gif_interval}) for GIF frames")
    else:
        # For custom gif_interval values
        ratio = gif_interval / save_interval if save_interval > 0 else 0
        print(f"Collecting GIF frames every {gif_interval} steps ({ratio:.1f}x the image interval)")
    
    # Keep track of saved states for animation
    saved_states = []
    saved_losses = []
    saved_steps = []
    
    def callback(step, state, loss):
        # Save current state image for monitoring
        if step % save_interval == 0 or step == 0:
            ax, fig = visualize_state(state, title=f"Step {step}, Loss: {loss:.6f}")
            plt.savefig(os.path.join(save_dir, f"state_{step:06d}.png"))
            plt.close(fig)  # Explicitly close the figure by reference
        
        # Only save for GIF at gif_interval steps or at the end
        # This reduces memory usage and makes GIF creation more efficient
        if step % gif_interval == 0 or step == 0:
            saved_states.append(state.clone().detach().cpu())
            saved_losses.append(loss)
            saved_steps.append(step)
    
    # Attach the saved data to the callback for later use
    callback.saved_states = saved_states
    callback.saved_losses = saved_losses
    callback.saved_steps = saved_steps
    callback.gif_interval = original_gif_interval  # Store the original gif_interval
    
    return callback 