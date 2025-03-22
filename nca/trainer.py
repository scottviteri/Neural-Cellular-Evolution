import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from nca.model import preprocess_target, postprocess_output

class LambdaSchedule:
    """
    Schedule for controlling the mixing between random initialization and previous state.
    Lambda=0 means fully random initialization (independent batches).
    Lambda=1 means fully using the previous state (traditional NCA behavior).
    """
    def __init__(self, schedule_type='constant', start_value=1.0, end_value=None, 
                 steps=None, warmup_steps=None):
        """
        Initialize a lambda schedule.
        
        Args:
            schedule_type: Type of schedule ('constant', 'linear', 'cosine', 'exponential', 'step')
            start_value: Initial lambda value
            end_value: Final lambda value (for non-constant schedules)
            steps: Total number of steps for the schedule
            warmup_steps: Number of warmup steps (for step schedule)
        """
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value if end_value is not None else start_value
        self.steps = steps or 10000
        self.warmup_steps = warmup_steps or 0
        
    def get_value(self, step):
        """
        Get lambda value at a given step.
        
        Args:
            step: Current step number
            
        Returns:
            Lambda value (between 0 and 1)
        """
        if self.schedule_type == 'constant':
            return self.start_value
            
        if step >= self.steps:
            return self.end_value
            
        progress = step / self.steps
        
        if self.schedule_type == 'linear':
            # Linear interpolation
            return self.start_value + (self.end_value - self.start_value) * progress
            
        elif self.schedule_type == 'cosine':
            # Cosine schedule (smoother transition)
            cos_progress = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            return self.end_value + (self.start_value - self.end_value) * cos_progress
            
        elif self.schedule_type == 'exponential':
            # Exponential decay
            decay_rate = (self.end_value / self.start_value) ** (1 / self.steps)
            return self.start_value * (decay_rate ** step)
            
        elif self.schedule_type == 'step':
            # Step schedule (constant until warmup, then jump to end value)
            if step < self.warmup_steps:
                return self.start_value
            else:
                return self.end_value
                
        return self.start_value  # Default fallback

class NCATrainer:
    """
    Trainer for Neural Cellular Automaton (NCA).
    Handles training and noise injection for a single long trajectory.
    """
    def __init__(
        self, 
        model,
        target,
        learning_rate=0.001,
        sigma_small=0.01,
        sigma_large=0.1,
        epsilon=0.001,
        device=None,
        debug=False,
        filter_improvements=False,
        lambda_schedule=None,
        random_init_sigma=0.1
    ):
        """
        Initialize the NCA trainer.
        
        Args:
            model: The NCA model to train
            target: Target image tensor (batch_size, channels, height, width)
            learning_rate: Learning rate for the optimizer
            sigma_small: Standard deviation for small constant noise
            sigma_large: Standard deviation for large noise spikes
            epsilon: Threshold for considering state "close" to target
            device: Device to run the model on (cuda or cpu), if None will use CUDA if available
            debug: Whether to collect debugging information during training
            filter_improvements: Whether to only update the model using samples that improved
            lambda_schedule: Schedule for mixing previous state with random initialization
            random_init_sigma: Standard deviation for random initialization
        """
        # Determine device 
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Move model and target to device
        self.model = model.to(self.device)
        self.original_target = target.to(self.device)
        
        # Always use logit transform for better gradient properties
        print("Using logit transform for target space")
        self.target = preprocess_target(self.original_target)
        
        # Store the number of RGB channels (assumed to be 3)
        # Important: We only use the first 3 channels (RGB) for loss computation
        # regardless of how many channels the model has in total
        self.rgb_channels = min(3, target.shape[1])  # Use all channels if less than 3
        print(f"Using {self.rgb_channels} channel(s) for loss computation")
        
        # Hyperparameters
        self.sigma_small = sigma_small
        self.sigma_large = sigma_large
        self.epsilon = epsilon
        self.debug = debug
        self.filter_improvements = filter_improvements
        self.random_init_sigma = random_init_sigma
        
        # Set up lambda schedule for mixing previous state with random initialization
        if lambda_schedule is None:
            # Default behavior: fully use previous state (traditional NCA)
            self.lambda_schedule = LambdaSchedule(schedule_type='constant', start_value=1.0)
        else:
            self.lambda_schedule = lambda_schedule
            
        # Print lambda schedule information
        print(f"Using lambda schedule: {self.lambda_schedule.schedule_type}, " 
              f"start={self.lambda_schedule.start_value}, end={self.lambda_schedule.end_value}")
        
        # Improvement tracking when filtering
        if filter_improvements:
            self.updates_skipped = 0  # Count of individual skipped samples
            self.total_samples = 0    # Count of total processed samples
            self.improvement_rates = []  # Track improvement rate per batch
            print("Filter improvements mode: Only updating model using samples that improved")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Batch size, channels, height, width from target
        self.batch_size, self.channels, self.height, self.width = target.shape
        
        # Training stats
        self.losses = []
        self.reset_count = 0
        self.reset_history = []  # Track which steps had resets
        self.lambda_values = []  # Track lambda values used
        
        # Debugging stats
        if debug:
            self.grad_norms = []
            self.pixel_means = []
            self.pixel_stds = []
            self.hidden_means = []
            self.hidden_stds = []
            # New: Track each RGB channel separately
            self.channel_means = [[] for _ in range(min(3, target.shape[1]))]
    
    def add_noise(self, state, sigma):
        """
        Add Gaussian noise to the state.
        
        Args:
            state: Current state tensor
            sigma: Standard deviation of the noise
            
        Returns:
            Noisy state tensor
        """
        # Ensure state is on the correct device and detached from previous computations
        state = state.detach().to(self.device)
        noise = torch.randn_like(state) * sigma
        return state + noise
    
    def initialize_state(self, init_type='random'):
        """
        Initialize a random state.
        
        Args:
            init_type: Type of initialization ('random', 'target', 'zeros', 'ones')
            
        Returns:
            Random state tensor
        """
        # Get channel count from the model, not the target
        model_channels = self.model.channels
        
        # Initialize the state based on the specified type
        if init_type == 'random':
            # Initialize with small random values
            state = torch.randn(
                self.batch_size, 
                model_channels,
                self.height, 
                self.width,
                device=self.device
            ) * 0.1
        elif init_type == 'target':
            # Initialize RGB channels with target and random for hidden channels
            state = torch.zeros(
                self.batch_size, 
                model_channels,
                self.height, 
                self.width,
                device=self.device
            )
            # Copy target RGB channels
            rgb_channels = min(3, self.target.shape[1])
            state[:, :rgb_channels] = self.target[:, :rgb_channels].clone()
            # Random initialization for hidden channels
            if model_channels > rgb_channels:
                state[:, rgb_channels:] = torch.randn(
                    self.batch_size, 
                    model_channels - rgb_channels,
                    self.height, 
                    self.width,
                    device=self.device
                ) * 0.1
        elif init_type == 'zeros':
            # Initialize with zeros
            state = torch.zeros(
                self.batch_size, 
                model_channels,
                self.height, 
                self.width,
                device=self.device
            )
        elif init_type == 'ones':
            # Initialize with ones
            state = torch.ones(
                self.batch_size, 
                model_channels,
                self.height, 
                self.width,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
        
        return state
    
    def get_gradient_stats(self):
        """
        Calculate gradient statistics for debugging.
        
        Returns:
            Dictionary with gradient statistics
        """
        if not self.debug:
            return {}
            
        stats = {}
        total_norm = 0.0
        param_norms = {}
        
        # Calculate gradient norms per parameter
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_norms[name] = param_norm
                total_norm += param_norm ** 2
                
        stats['total_norm'] = total_norm ** 0.5
        stats['param_norms'] = param_norms
        
        return stats
    
    def generate_random_state(self):
        """
        Generate a random state with the same shape as the model's state.
        
        Returns:
            Random state tensor
        """
        model_channels = self.model.channels
        return torch.randn(
            self.batch_size, 
            model_channels,
            self.height, 
            self.width,
            device=self.device
        ) * self.random_init_sigma
        
    def mix_states(self, previous_state, random_state, lambda_val):
        """
        Mix previous state with random state according to lambda value.
        
        Args:
            previous_state: Previous state tensor
            random_state: Random state tensor
            lambda_val: Mixing coefficient (0=fully random, 1=fully previous)
            
        Returns:
            Mixed state tensor
        """
        return lambda_val * previous_state + (1 - lambda_val) * random_state
    
    def train_step(self, state, step):
        """
        Perform a single training step.
        
        Args:
            state: Current state tensor
            step: Current step number
            
        Returns:
            Tuple of (new_state, loss, reset, mse_before, improvement)
            where reset is a boolean indicating if a noise spike was applied,
            mse_before is the MSE before the update, and improvement is the change in MSE
        """
        # IMPORTANT: Detach state to prevent gradient flow between steps
        state = state.detach()
        
        # Get lambda value for this step
        lambda_val = self.lambda_schedule.get_value(step)
        self.lambda_values.append(lambda_val)
        
        # Generate random state
        random_state = self.generate_random_state()
        
        # Mix previous state with random state
        mixed_state = self.mix_states(state, random_state, lambda_val)
        
        # Add small constant noise to the mixed state
        noisy_state = self.add_noise(mixed_state, self.sigma_small)
        
        # Step 1: Forward pass with noisy_state 
        # Enable gradients for the current step only
        noisy_state.requires_grad_(True)
        next_state = self.model(noisy_state)
        
        # Get RGB channels for both states
        rgb_noisy_state = noisy_state[:, :self.rgb_channels]
        rgb_next_state = next_state[:, :self.rgb_channels]
        rgb_target = self.target[:, :self.rgb_channels]
        
        # Calculate per-sample MSE before the update (without reduction)
        with torch.no_grad():
            # Calculate MSE for each batch sample individually
            # Shape: [batch_size]
            mse_before_per_sample = torch.mean(
                (rgb_noisy_state - rgb_target)**2, 
                dim=[1, 2, 3]  # Average across channels, height, width
            )
            # Calculate average MSE before (for reporting)
            mse_before = mse_before_per_sample.mean().item()
        
        # Calculate per-sample MSE after the update (without reduction)
        # Shape: [batch_size]
        mse_after_per_sample = torch.mean(
            (rgb_next_state - rgb_target)**2, 
            dim=[1, 2, 3]  # Average across channels, height, width
        )
        # Calculate average MSE after (for reporting)
        mse_after = mse_after_per_sample.mean().item()
        
        # Determine which samples improved (lower MSE)
        # Shape: [batch_size]
        improved_mask = mse_after_per_sample < mse_before_per_sample
        
        # How many samples improved
        num_improved = improved_mask.sum().item()
        improvement_rate = num_improved / self.batch_size
        
        # Track improvement statistics when filtering
        if self.filter_improvements:
            # Update the count of skipped samples
            self.updates_skipped += (self.batch_size - num_improved)
            # Track the improvement rates
            if hasattr(self, 'improvement_rates'):
                self.improvement_rates.append(improvement_rate)
        
        # Compute the masked loss (only for improved samples)
        if self.filter_improvements and num_improved > 0:
            # Create a filtered loss using only the improved samples
            # Either: weight the loss by the improved mask
            improved_weights = improved_mask.float()
            # Normalize by the number of improved samples
            improved_weights = improved_weights / (num_improved if num_improved > 0 else 1)
            
            # Weighted MSE loss
            loss = torch.sum(mse_after_per_sample * improved_weights)
        else:
            # If not filtering or no samples improved, use standard MSE
            loss = mse_after_per_sample.mean()
        
        # Compute gradients and update model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Calculate overall improvement
        improvement = mse_before - mse_after
        
        # Collect debugging info if enabled (before optimizer step)
        if self.debug:
            # Get gradient statistics
            grad_stats = self.get_gradient_stats()
            if grad_stats:
                self.grad_norms.append(grad_stats['total_norm'])
            
            # Track pixel statistics (RGB channels)
            with torch.no_grad():
                self.pixel_means.append(rgb_next_state.mean().item())
                self.pixel_stds.append(rgb_next_state.std().item())
                
                # Track individual RGB channel means
                for c in range(min(3, self.rgb_channels)):
                    channel_mean = rgb_next_state[:, c].mean().item()
                    self.channel_means[c].append(channel_mean)
                
                # Track hidden state statistics if there are hidden channels
                if next_state.shape[1] > self.rgb_channels:
                    hidden_state = next_state[:, self.rgb_channels:]
                    self.hidden_means.append(hidden_state.mean().item())
                    self.hidden_stds.append(hidden_state.std().item())
        
        # Update model only if conditions are met
        update_model = True  # Default behavior
        
        if self.filter_improvements:
            # Only update if at least one sample improved
            update_model = num_improved > 0
        
        if update_model:
            self.optimizer.step()
        
        # For the state transition, we don't want gradients to flow through
        with torch.no_grad():
            # Check if state is close to target
            reset = False
            if mse_after < self.epsilon:
                # Apply large noise spike
                next_state = self.add_noise(next_state, self.sigma_large)
                reset = True
                self.reset_count += 1
            else:
                # Detach next_state to prevent gradient propagation between steps
                next_state = next_state.detach()
        
        return next_state, mse_after, reset, mse_before, improvement, num_improved
    
    def get_displayable_state(self, state):
        """
        Convert state to displayable format (range [0,1]).
        
        Args:
            state: Current state tensor
            
        Returns:
            State tensor in [0,1] range for visualization
        """
        # If we have more than RGB channels, only display the RGB channels
        if state.shape[1] > 3:
            state_rgb = state[:, :3].clone()
            return postprocess_output(state_rgb)
        return postprocess_output(state)
    
    def train(self, steps=10000, log_interval=100, callback=None, init_type='random'):
        """
        Train the NCA for a specified number of steps.
        
        Args:
            steps: Number of training steps
            log_interval: Interval for logging detailed progress to console
            callback: Optional callback function called at each step
                    with (step, state, loss) as arguments. The callback is responsible
                    for filtering based on intervals for visualization saving.
            init_type: Type of initialization for the state ('random', 'target', 'zeros', 'ones')
        
        Returns:
            Dictionary with training statistics
        """
        state = self.initialize_state(init_type)
        progress_bar = tqdm(range(steps), desc="Training NCA")
        
        # Additional stats to track
        self.mse_before_list = []
        self.improvement_list = []
        self.samples_improved_list = []  # New: track samples improved per step
        
        for step in progress_bar:
            # Perform training step
            state, loss, reset, mse_before, improvement, num_improved = self.train_step(state, step)
            
            # Update total samples processed
            if self.filter_improvements:
                self.total_samples += self.batch_size
                
            # Store loss and reset info
            self.losses.append(loss)
            self.reset_history.append(reset)
            self.mse_before_list.append(mse_before)
            self.improvement_list.append(improvement)
            self.samples_improved_list.append(num_improved)
            
            # Update progress bar with additional info when filtering updates
            postfix = {"loss": f"{loss:.6f}", "resets": self.reset_count}
            if self.filter_improvements:
                # Include percentage of improved samples in this batch
                improved_pct = (num_improved / self.batch_size) * 100
                # Include overall improvement rate
                total_improved = self.total_samples - self.updates_skipped
                overall_rate = (total_improved / self.total_samples) * 100 if self.total_samples > 0 else 0
                
                postfix["improved"] = f"{num_improved}/{self.batch_size} ({improved_pct:.1f}%)"
                postfix["overall"] = f"{total_improved}/{self.total_samples} ({overall_rate:.1f}%)"
                
            progress_bar.set_postfix(postfix)
            
            # Only print detailed info at log_interval
            if step % log_interval == 0 or step == steps - 1:
                # Print improvement information at log intervals
                improvement_info = f"Step {step}: MSE Before = {mse_before:.6f}, MSE After = {loss:.6f}, Improvement = {improvement:.6f}"
                if self.filter_improvements:
                    improved_pct = (num_improved / self.batch_size) * 100
                    total_improved = self.total_samples - self.updates_skipped
                    overall_rate = (total_improved / self.total_samples) * 100 if self.total_samples > 0 else 0
                    improvement_info += f", Samples Improved = {num_improved}/{self.batch_size} ({improved_pct:.1f}%)"
                    improvement_info += f", Overall = {total_improved}/{self.total_samples} ({overall_rate:.1f}%)"
                print(improvement_info)
            
            # Always call callback if provided - let the callback decide when to save based on its own intervals
            if callback is not None:
                # Convert state to displayable format for visualization
                displayable_state = self.get_displayable_state(state)
                callback(step, displayable_state, loss)
        
        # Get final displayable state
        final_displayable_state = self.get_displayable_state(state)
        
        # Prepare results with additional stats for filtering
        results = {
            "losses": self.losses,
            "reset_count": self.reset_count,
            "reset_history": self.reset_history,
            "final_state": final_displayable_state,
            "final_loss": self.losses[-1],
            "mse_before_list": self.mse_before_list,
            "improvement_list": self.improvement_list,
            "samples_improved_list": self.samples_improved_list,
            "lambda_values": self.lambda_values
        }
        
        if self.filter_improvements:
            total_improved = self.total_samples - self.updates_skipped
            results["updates_skipped"] = self.updates_skipped
            results["total_samples"] = self.total_samples
            results["samples_improved"] = total_improved
            results["improvement_rate"] = total_improved / self.total_samples * 100 if self.total_samples > 0 else 0
            if hasattr(self, 'improvement_rates'):
                results["improvement_rates"] = self.improvement_rates
            
        return results 