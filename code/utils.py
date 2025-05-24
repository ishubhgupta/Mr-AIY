"""
Utility functions for the AI Vision Suite
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
from tqdm import tqdm
import json
from typing import Dict, Any, Optional

def ensure_dir(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_image(image_path: str, target_size: Optional[tuple] = None) -> np.ndarray:
    """
    Load and preprocess an image file
    
    Args:
        image_path: Path to the image file
        target_size: Optional tuple (width, height) to resize image
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        image = Image.open(image_path)
        
        if target_size:
            image = image.resize(target_size)
            
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        return image_array
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model input.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size for resizing
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Normalize to [0, 1]
    image_np = image_np / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return image_tensor

def plot_to_base64(plt_figure):
    """
    Convert a matplotlib figure to a base64 encoded string.
    
    Args:
        plt_figure (matplotlib.figure.Figure): Matplotlib figure
        
    Returns:
        str: Base64 encoded string of the image
    """
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

def get_model_info(model):
    """
    Get information about a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        dict: Model information
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = {
        "num_parameters": num_params,
        "modules": str(model)
    }
    return model_info

def save_model(model, path, metadata=None):
    """
    Save a PyTorch model with metadata.
    
    Args:
        model (torch.nn.Module): PyTorch model to save
        path (str): Path to save the model
        metadata (dict, optional): Additional metadata to save with the model
        
    Returns:
        str: Path to the saved model
    """
    ensure_dir(os.path.dirname(path))
    
    # Save model state
    model_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    # Add metadata if provided
    if metadata:
        model_dict['metadata'] = metadata
        
    torch.save(model_dict, path)
    return path

def load_saved_model(model, path):
    """
    Load a saved PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model instance to load into
        path (str): Path to the saved model
        
    Returns:
        tuple: (model, metadata)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    metadata = checkpoint.get('metadata', {})
    return model, metadata

def prepare_tabular_data(data_path, target_column=None):
    """
    Load and prepare tabular data.
    
    Args:
        data_path (str): Path to the data file
        target_column (str, optional): Name of the target column
        
    Returns:
        tuple: (X, y) or X if target_column is None
    """
    # Load data based on file extension
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV, Excel, or JSON.")
    
    # Handle missing values
    df = df.dropna()
    
    # Split into features and target if target_column is provided
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    
    return df

def count_images_in_directory(directory, extensions=['.jpg', '.jpeg', '.png']):
    """
    Count the number of images in a directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory
        extensions (list): List of valid image extensions
        
    Returns:
        dict: Dictionary with class names and image counts
    """
    class_counts = {}
    
    for root, dirs, files in os.walk(directory):
        class_name = os.path.basename(root)
        if class_name != os.path.basename(directory):  # Skip the main directory
            image_count = sum(1 for f in files if os.path.splitext(f)[1].lower() in extensions)
            if image_count > 0:
                class_counts[class_name] = image_count
    
    return class_counts

def create_timestamp():
    """
    Create a formatted timestamp for filenames.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        return f"{int(m)}m {int(s)}s"
    else:
        return f"{s:.2f}s"

def save_model_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """
    Save model metadata to a JSON file
    
    Args:
        model_path: Path to the model file
        metadata: Dictionary containing model metadata
    """
    try:
        metadata_path = model_path.replace('.pkl', '_metadata.json').replace('.h5', '_metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
    except Exception as e:
        raise ValueError(f"Error saving metadata: {str(e)}")

def load_model_metadata(model_path: str) -> Dict[str, Any]:
    """
    Load model metadata from a JSON file
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model metadata
    """
    try:
        metadata_path = model_path.replace('.pkl', '_metadata.json').replace('.h5', '_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {}
            
    except Exception as e:
        raise ValueError(f"Error loading metadata: {str(e)}")

def get_storage_info() -> Dict[str, float]:
    """
    Get storage information for the current directory
    
    Returns:
        Dictionary with storage information in MB
    """
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        return {
            'used': total_size / (1024 * 1024)  # Convert to MB
        }
    except Exception:
        return {'used': 0.0}
