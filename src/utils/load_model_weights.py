import torch
import os

def load_model_weights(model, weights_dir, device):
    """
    Load the weights of the model from a specified directory.

    Args:
        model (nn.Module): The model instance to load weights into.
        weights_dir (str): The directory containing the file with model weights.
        device (torch.device): The device to map the model and weights onto.

    Returns:
        nn.Module: The model with loaded weights.
    """
    
    weights_path_file = os.path.join(weights_dir, "best_model.pth")
    
    if not os.path.exists(weights_path_file):
        raise FileNotFoundError(f"Weights file not found at {weights_path_file}")
    
    try:
        # Load the state dictionary directly since that's what was saved.
        state_dict = torch.load(weights_path_file, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully from {weights_path_file}")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {e}")
    
    return model
