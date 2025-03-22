import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from nca.model import NCA, preprocess_target, postprocess_output
from nca.trainer import NCATrainer, LambdaSchedule
from nca.visualization import (
    visualize_state, 
    plot_loss, 
    create_animation,
    create_training_callback
)


def load_target_image(path, size=(32, 32)):
    """
    Load and preprocess target image.
    
    Args:
        path: Path to the target image
        size: Tuple of (height, width) to resize image to
        
    Returns:
        PyTorch tensor of shape (1, channels, height, width)
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    # Convert to PyTorch tensor and add batch dimension
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, height, width)
    
    return img_tensor


def create_simple_target(type="solid", size=(32, 32), channels=3, batch_size=1):
    """
    Create a simple target image.
    
    Args:
        type: Type of simple target ("solid", "gradient", "checkerboard", "circle")
        size: Tuple of (height, width)
        channels: Number of channels
        batch_size: Number of times to repeat the target
        
    Returns:
        PyTorch tensor of shape (batch_size, channels, height, width)
    """
    height, width = size
    
    if type == "solid":
        # Solid color (red)
        target = torch.zeros(batch_size, channels, height, width)
        target[:, 0, :, :] = 1.0  # Red channel
        
    elif type == "gradient":
        # Horizontal gradient
        x = torch.linspace(0, 1, width)
        gradient = x.repeat(height, 1)
        target = torch.zeros(batch_size, channels, height, width)
        # Set different gradient directions for each channel
        target[:, 0, :, :] = gradient
        if channels > 1:
            target[:, 1, :, :] = gradient.t()  # Vertical gradient
        if channels > 2:
            target[:, 2, :, :] = 0.5  # Constant value
            
    elif type == "checkerboard":
        # Checkerboard pattern
        x = torch.arange(width).reshape(1, width).repeat(height, 1)
        y = torch.arange(height).reshape(height, 1).repeat(1, width)
        pattern = ((x + y) % 2 == 0).float()
        target = torch.zeros(batch_size, channels, height, width)
        
        # Set different patterns for different channels
        target[:, 0, :, :] = pattern
        if channels > 1:
            target[:, 1, :, :] = 1 - pattern  # Inverse
        if channels > 2:
            target[:, 2, :, :] = (x % 2 == 0).float()  # Stripes
            
    elif type == "circle":
        # Circle in the middle
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        
        x = torch.arange(width).reshape(1, width).repeat(height, 1)
        y = torch.arange(height).reshape(height, 1).repeat(1, width)
        
        # Distance from center
        dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        circle = (dist <= radius).float()
        
        target = torch.zeros(batch_size, channels, height, width)
        if channels == 1:
            target[:, 0, :, :] = circle
        else:
            # RGB circle
            target[:, 0, :, :] = circle  # Red
            if channels > 1:
                target[:, 1, :, :] = 0.0  # Green
            if channels > 2:
                target[:, 2, :, :] = 0.0  # Blue
    
    else:
        raise ValueError(f"Unknown target type: {type}")
    
    return target


def create_lambda_schedule(args):
    """
    Create a lambda schedule based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        LambdaSchedule object
    """
    # Default to constant schedule with lambda=1.0 (standard NCA behavior)
    if not hasattr(args, 'lambda_schedule') or args.lambda_schedule == 'constant':
        lambda_val = getattr(args, 'lambda_value', 1.0)
        return LambdaSchedule(
            schedule_type='constant', 
            start_value=lambda_val
        )
    
    # Get common parameters
    lambda_start = getattr(args, 'lambda_start', 1.0)
    lambda_end = getattr(args, 'lambda_end', 0.0)
    steps = getattr(args, 'steps', 10000)
    
    if args.lambda_schedule == 'independent':
        # Independent batches (lambda=0)
        return LambdaSchedule(
            schedule_type='constant', 
            start_value=0.0
        )
    
    elif args.lambda_schedule == 'linear':
        return LambdaSchedule(
            schedule_type='linear',
            start_value=lambda_start,
            end_value=lambda_end,
            steps=steps
        )
    
    elif args.lambda_schedule == 'cosine':
        return LambdaSchedule(
            schedule_type='cosine',
            start_value=lambda_start,
            end_value=lambda_end,
            steps=steps
        )
    
    elif args.lambda_schedule == 'exponential':
        return LambdaSchedule(
            schedule_type='exponential',
            start_value=lambda_start,
            end_value=lambda_end,
            steps=steps
        )
    
    elif args.lambda_schedule == 'step':
        warmup_steps = getattr(args, 'warmup_steps', steps // 2)
        return LambdaSchedule(
            schedule_type='step',
            start_value=lambda_start,
            end_value=lambda_end,
            steps=steps,
            warmup_steps=warmup_steps
        )
    
    # Default fallback
    return LambdaSchedule(
        schedule_type='constant',
        start_value=1.0
    ) 


def run_training(args):
    """
    Run NCA training with given arguments.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Always use 'results' as output directory regardless of args
    output_dir = Path('results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get RGB channels for target (default to 3 or user-specified)
    rgb_channels = getattr(args, 'rgb_channels', 3)
    
    # Load or create target
    if args.target_image:
        # For image targets, we'll create a batch by repeating the image
        single_target = load_target_image(args.target_image, size=(args.size, args.size))
        target = single_target.repeat(args.batch_size, 1, 1, 1)  # Repeat to match batch size
        print(f"Loaded target from {args.target_image} and replicated to batch size {args.batch_size}")
    else:
        target = create_simple_target(
            args.target_type, 
            size=(args.size, args.size), 
            channels=rgb_channels,  # Use rgb_channels for target
            batch_size=args.batch_size
        )
        print(f"Created {args.target_type} target of size {args.size}x{args.size} with batch size {args.batch_size}")
    
    # Save target image (just the first example in the batch)
    fig = plt.figure(figsize=(5, 5))
    img = target[0].permute(1, 2, 0).numpy()
    plt.imshow(np.clip(img, 0, 1))
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "target.png"))
    plt.close(fig)
    
    # Create model with specified number of channels
    model = NCA(
        channels=args.channels,  # Total channels for the model
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_nonlinearity=args.use_nonlinearity,
        nonlin_type=args.nonlin_type
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    nonlin_str = f"with {args.nonlin_type} nonlinearity" if args.use_nonlinearity else "without nonlinearity"
    print(f"Created NCA model with {total_params} parameters, {nonlin_str}, {args.num_blocks} transformer block(s)")
    print(f"Using {args.channels} channels for model state (including {min(rgb_channels, 3)} RGB channels)")
    
    # Set up lambda schedule based on arguments
    lambda_schedule = create_lambda_schedule(args)
    
    # Create trainer
    trainer = NCATrainer(
        model=model,
        target=target,
        learning_rate=args.learning_rate,
        sigma_small=args.noise_small,
        sigma_large=args.noise_large,
        device=device,
        debug=args.debug,
        filter_improvements=args.filter_improvements,
        lambda_schedule=lambda_schedule,
        random_init_sigma=args.random_init_sigma
    )
    
    # Prepare for visualizations
    saved_states = []
    saved_losses = []
    saved_steps = []
    plot_update_interval = args.image_interval  # Separate interval for updating debug plots
    
    # Create directory for saving visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up training callback that handles both state saving and GIF updating
    print(f"Setting up visualization with image_interval={args.image_interval}, gif_interval={args.gif_interval}")
    callback = create_training_callback(
        save_dir=output_dir,
        save_interval=args.image_interval,  # Use image_interval for state image saving
        gif_interval=args.gif_interval
    )
    
    # Get initialization type
    init_type = getattr(args, 'init_type', 'random')
    print(f"Initializing state with '{init_type}' strategy")
    
    # Run training
    print(f"Starting training for {args.steps} steps with batch size {args.batch_size}...")
    # Run training - note that the callback is responsible for applying plot_interval filters
    results = trainer.train(
        steps=args.steps,
        log_interval=args.log_interval,  # This is just for console logging, not related to visualization
        callback=callback,
        init_type=init_type
    )
    
    # Save final state
    fig = plt.figure(figsize=(10, 5))
    axes, _ = visualize_state(
        results["final_state"], 
        target=target[0:1],
        title=f"Final State (Loss: {results['final_loss']:.6f})"
    )
    plt.savefig(os.path.join(output_dir, "final_state.png"))
    plt.close(fig)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    # Save loss plot
    fig = plot_loss(results["losses"])
    plt.title("Loss History")
    plt.savefig(os.path.join(output_dir, "loss_history.png"))
    plt.close(fig)
    
    # If we have lambda values, plot them
    if results.get("lambda_values"):
        fig = plt.figure(figsize=(10, 5))
        plt.plot(results["lambda_values"])
        plt.title("Lambda Values (State Mixing Coefficients)")
        plt.xlabel("Step")
        plt.ylabel("Lambda")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "lambda_values.png"))
        plt.close(fig)
    
    # Plot final debugging information if available
    if trainer.debug:
        # Plot gradient norms
        if trainer.grad_norms:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(trainer.grad_norms)
            plt.title("Gradient Norm History")
            plt.xlabel("Step")
            plt.ylabel("Gradient Norm")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "gradient_norms.png"))
            plt.close(fig)
        
        # Plot MSE before/after and improvement
        if hasattr(trainer, 'mse_before_list') and trainer.mse_before_list:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(trainer.mse_before_list, label="MSE Before")
            plt.plot(trainer.losses, label="MSE After")
            plt.title("MSE Before vs After")
            plt.xlabel("Step")
            plt.ylabel("MSE")
            plt.yscale("log")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
            plt.close(fig)
            
            # Plot improvement (positive is good, negative is bad)
            fig = plt.figure(figsize=(10, 5))
            plt.plot(trainer.improvement_list)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title("MSE Improvement")
            plt.xlabel("Step")
            plt.ylabel("Improvement (Positive is better)")
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "mse_improvement.png"))
            plt.close(fig)
            
            # Plot number of improved samples per batch
            if hasattr(trainer, 'samples_improved_list') and trainer.samples_improved_list:
                fig = plt.figure(figsize=(10, 5))
                plt.plot(trainer.samples_improved_list)
                plt.axhline(y=trainer.batch_size, color='g', linestyle='--', label=f"Batch Size ({trainer.batch_size})")
                plt.axhline(y=trainer.batch_size/2, color='y', linestyle='--', label="50% of Batch")
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title("Samples Improved Per Batch")
                plt.xlabel("Step")
                plt.ylabel("Number of Samples")
                plt.ylim(0, trainer.batch_size)
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "samples_improved.png"))
                plt.close(fig)
        
        # Plot pixel statistics
        if trainer.pixel_means and trainer.pixel_stds:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(trainer.pixel_means, label="Mean")
            plt.plot(trainer.pixel_stds, label="Std")
            plt.title("RGB Pixel Statistics")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "pixel_statistics.png"))
            plt.close(fig)
        
        # Plot individual RGB channel means
        if hasattr(trainer, 'channel_means') and trainer.channel_means:
            fig = plt.figure(figsize=(10, 5))
            colors = ['r', 'g', 'b']
            labels = ['Red Channel', 'Green Channel', 'Blue Channel']
            
            for i, means in enumerate(trainer.channel_means):
                if means:  # Only plot if we have data
                    plt.plot(means, color=colors[i], label=labels[i])
                    
            plt.title("Individual RGB Channel Means")
            plt.xlabel("Step")
            plt.ylabel("Channel Mean")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "rgb_channel_means.png"))
            plt.close(fig)
        
        # Plot hidden state statistics
        if trainer.hidden_means and trainer.hidden_stds:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(trainer.hidden_means, label="Mean")
            plt.plot(trainer.hidden_stds, label="Std")
            plt.title("Hidden State Statistics")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "hidden_statistics.png"))
            plt.close(fig)
    
    # Create and save the training animation as GIF
    if not args.disable_gif:  # Only create animation if GIF generation is enabled
        print("Creating training animation...")
        if hasattr(callback, 'saved_states') and len(callback.saved_states) > 0:
            # Check if we have enough frames for an animation
            if len(callback.saved_states) > 1:
                # Create the animation
                animation = create_animation(
                    states=callback.saved_states,
                    steps=callback.saved_steps,
                    losses=callback.saved_losses,
                    interval=200  # 200ms between frames (5 fps)
                )
                
                # Save the animation as a GIF
                try:
                    from matplotlib.animation import PillowWriter
                    gif_path = os.path.join(output_dir, "training_animation.gif")
                    animation.save(gif_path, writer=PillowWriter(fps=5))
                    print(f"Training animation saved to {gif_path} with {len(callback.saved_states)} frames")
                    print(f"GIF frames were collected every {args.gif_interval} steps (10x the image interval)")
                except ImportError:
                    print("Could not save GIF: Pillow not installed. Install with: pip install pillow")
                except Exception as e:
                    print(f"Error saving GIF animation: {e}")
            else:
                print(f"Only {len(callback.saved_states)} frames collected - need at least 2 for animation.")
                print(f"Try reducing --image-interval (currently {args.image_interval}) or running for more steps.")
        else:
            print("No frames collected for animation.")
    else:
        print("GIF animation generation disabled (--disable-gif flag was used)")
    
    print(f"Training complete! Results saved to {output_dir}")
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Total resets: {results['reset_count']}")


def run_quick_demo():
    """Run a quick demo of the NCA with fixed parameters."""
    import argparse
    
    # Create default arguments
    args = argparse.Namespace(
        target_type='circle',
        target_image=None,
        size=32,
        channels=16,  # Total channels for the model
        rgb_channels=3,  # RGB channels for target
        batch_size=8,
        steps=5000,
        learning_rate=0.001,
        noise_small=0.01,
        noise_large=0.1,
        epsilon=0.001,
        log_interval=100,
        output_dir='results',  # Always use 'results' folder
        debug=True,
        cpu=False,
        hidden_dim=16,
        num_blocks=2,
        mlp_ratio=4.0,
        dropout=0.0,
        use_nonlinearity=True,
        nonlin_type='relu',
        init_type='target',
        filter_improvements=True,
        lambda_schedule='constant',
        lambda_value=1.0,
        random_init_sigma=0.1,
        image_interval=100,
        disable_gif=False
    )
    
    # Calculate gif_interval as 10x the image_interval
    args.gif_interval = args.image_interval * 10
    
    run_training(args)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Neural Cellular Automaton")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=16,
                        help="Hidden dimension for attention mechanism")
    parser.add_argument("--num-blocks", type=int, default=1,
                        help="Number of transformer blocks to use")
    parser.add_argument("--mlp-ratio", type=float, default=4.0,
                        help="Ratio for MLP hidden dimension compared to input dimension")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability")
    parser.add_argument("--use-nonlinearity", action="store_true",
                        help="Whether to use a nonlinearity after the attention layer")
    parser.add_argument("--nonlin-type", type=str, default="tanh", choices=["tanh", "relu", "leaky_relu"],
                        help="Type of nonlinearity to use")
    
    # Target parameters
    parser.add_argument("--target-image", type=str, default=None,
                        help="Path to target image (if not provided, a simple target will be created)")
    parser.add_argument("--target-type", type=str, default="circle",
                        choices=["solid", "gradient", "checkerboard", "circle"],
                        help="Type of simple target to create (if --target-image not provided)")
    parser.add_argument("--size", type=int, default=32,
                        help="Size of target image (square)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training (default: 8)")
    
    # Channel parameters
    parser.add_argument('--channels', type=int, default=16,
                      help="Number of channels for the model (RGB + hidden)")
    parser.add_argument('--rgb-channels', type=int, default=3,
                      help="Number of RGB channels for the target (usually 3)")
    
    # Training parameters
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps (default: 10000)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--noise-small", type=float, default=0.01,
                        help="Standard deviation for small constant noise")
    parser.add_argument("--noise-large", type=float, default=0.1,
                        help="Standard deviation for large noise spikes")
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="Threshold for considering state close to target")
    parser.add_argument("--init-type", type=str, default="random",
                        choices=["random", "target", "zeros", "ones"],
                        help="Type of initialization for the state")
    parser.add_argument("--filter-improvements", action="store_true",
                        help="Only update the model when MSE improves")
    
    # Output parameters - removed output_dir as we always use 'results'
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Interval for console logging/output")
    parser.add_argument("--image-interval", type=int, default=100,
                        help="Interval for saving state images during training (0 to disable)")
    
    # Run mode
    parser.add_argument("--demo", action="store_true",
                        help="Run quick demo with default parameters")
    
    # Other parameters
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug information collection")
    
    # Add lambda schedule arguments
    parser.add_argument('--lambda-schedule', type=str, default='constant',
                      choices=['constant', 'independent', 'linear', 'cosine', 'exponential', 'step'],
                      help='Lambda schedule type for mixing previous and random states')
    parser.add_argument('--lambda-value', type=float, default=1.0,
                      help='Lambda value for constant schedule (1.0=fully use previous state, 0.0=fully random)')
    parser.add_argument('--lambda-start', type=float, default=1.0,
                      help='Starting lambda value for non-constant schedules')
    parser.add_argument('--lambda-end', type=float, default=0.0,
                      help='Ending lambda value for non-constant schedules')
    parser.add_argument('--warmup-steps', type=int, default=None,
                      help='Warmup steps for step schedule')
    parser.add_argument('--random-init-sigma', type=float, default=0.1,
                      help='Standard deviation for random initialization component')
    
    # GIF creation setting
    parser.add_argument('--disable-gif', action="store_true",
                      help='Disable GIF animation creation')
    
    args = parser.parse_args()
    
    # Always set output_dir to 'results' regardless of command line arguments
    args.output_dir = 'results'
    
    # Handle parameter name compatibility for legacy args
    if hasattr(args, 'lr') and not hasattr(args, 'learning_rate'):
        args.learning_rate = args.lr
    
    if hasattr(args, 'sigma_small') and not hasattr(args, 'noise_small'):
        args.noise_small = args.sigma_small
        
    if hasattr(args, 'sigma_large') and not hasattr(args, 'noise_large'):
        args.noise_large = args.sigma_large
        
    # For backward compatibility with older code that uses plot_interval or plot_update_interval
    if hasattr(args, 'plot_update_interval') and not hasattr(args, 'image_interval'):
        args.image_interval = args.plot_update_interval
    elif hasattr(args, 'plot_interval') and not hasattr(args, 'image_interval'):
        args.image_interval = args.plot_interval
        
    # Calculate gif_interval as 10x the image_interval, unless disabled
    if args.disable_gif:
        args.gif_interval = 0  # Disable GIF creation
    else:
        args.gif_interval = args.image_interval * 10  # GIF interval is 10x image interval
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.demo:
        run_quick_demo()
    else:
        run_training(args) 