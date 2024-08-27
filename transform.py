import torch
import torch.nn.functional as F


class LogitsSmoothingTransform(torch.nn.Module):
    def __init__(self, max_smooth_factor = 0.2, min_smooth_factor = 0.01, labels_dim=-1, by_pass = False, p=0.5):
        super(LogitsSmoothingTransform, self).__init__()  # Initialize the parent class
        self.max_smooth_factor = max_smooth_factor
        self.min_smooth_factor = min_smooth_factor
        self.labels_dim = labels_dim
        self.by_pass = by_pass
        self.p = p # probability to apply transform
        
    def label_smoothing_log_probs(self, log_probs):
        """
        Apply label smoothing to log probabilities.
        
        Args:
        - log_probs: Tensor of shape (batch_size, seq_length, num_classes) containing log probabilities.
        - smoothing: The smoothing factor (a float, e.g., 0.1)
        
        Returns:
        - smoothed_log_probs: Tensor of the same shape as log_probs, with smoothed log probabilities.
        """
        num_classes = log_probs.size(-1)

        random_float = torch.rand((1, 1)) * (self.max_smooth_factor - self.min_smooth_factor) + self.min_smooth_factor
        # Remove extra dimensions and get the value as a scalar
        smoothing = random_float.item()
        
        # Convert log_probs to probabilities
        probs = log_probs.exp()
        # Apply the label smoothing formula to probabilities
        smoothed_probs = (1.0 - smoothing) * probs + smoothing / num_classes
        # Convert smoothed probabilities back to log probabilities
        smoothed_log_probs = smoothed_probs.log()
        
        return smoothed_log_probs

    def forward(self, x):
        if torch.rand((1, 1)) > self.p:
            return x
        return self.label_smoothing_log_probs(x) if not self.by_pass else x
    
class LogitsRandomMaxTransform(torch.nn.Module):
    def __init__(self, randoms_factor = 0.2, labels_dim=-1, by_pass = False, p=0.5):
        super(LogitsRandomMaxTransform, self).__init__()  # Initialize the parent class
        self.randoms_factor = randoms_factor
        self.labels_dim = labels_dim
        self.by_pass = by_pass
        self.p = p # probability to apply transform
        
    def modify_max_probabilities(self, tensor):
        """
        Modify some of the maximum probabilities in the tensor to random classes.
        
        Args:
            tensor (torch.Tensor): Input tensor of shape [batch_size, seq_length, num_classes].
            num_changes (int): Number of maximum probabilities to change per sequence.
            min_class (int): Minimum class index for random replacement.
            max_class (int): Maximum class index for random replacement.
        
        Returns:
            torch.Tensor: Modified tensor.
        """
        # Ensure tensor is on CPU for indexing operations
        device = tensor.device

        # Create a copy of the tensor to modify
        modified_tensor = tensor.clone().cpu()

        batch_size, seq_length, num_classes = tensor.shape
        
        num_changes = int(self.randoms_factor * seq_length)

        # Generate random indices for changes
        for i in range(batch_size):
            # Get the index of the maximum probability in the last dimension
            max_indices = tensor[i].argmax(dim=self.labels_dim)
            
            # Choose random indices along the sequence length to change
            indices_to_change = torch.randint(0, seq_length, (num_changes,), dtype=torch.long)
            
            for idx in indices_to_change:
                # Get the class with the maximum probability
                max_class_idx = max_indices[idx].item()
                max_class_value = modified_tensor[i, idx, max_class_idx].item()
                
                # Generate a random class index different from the original max class index
                random_class_idx = torch.randint(0, num_classes, (1,)).item()
                random_class_value = modified_tensor[i, idx, random_class_idx].item()
                
                # Replace the maximum probability with a new random class probability
                modified_tensor[i, idx, max_class_idx] = random_class_value  # Set original max class to 0
                modified_tensor[i, idx, random_class_idx] = max_class_value  # Set new random class to 1
        
        return modified_tensor.to(device)
        
    def forward(self, x):
        if torch.rand((1, 1)) > self.p:
            return x
        return self.modify_max_probabilities(x) if not self.by_pass else x