import os
import requests
import torch
import numpy as np
import folder_paths
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

# Model download URLs
MODEL_URLS = {
    "multitask_large": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx",
    "depth_base": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/depth-model-vitb16_384.onnx",
    "depth_large": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/depth-model-vitl16_384.onnx",
    "foreground_base": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitb16_384.onnx",
    "foreground_large": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitl16_384.onnx",
    "normal_base": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/normal-model-vitb16_384.onnx",
    "normal_large": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/normal-model-vitl16_384.onnx",
}

def get_david_models_path() -> str:
    """Get the DAViD models directory path."""
    models_dir = folder_paths.models_dir
    david_dir = os.path.join(models_dir, "DAViD")
    os.makedirs(david_dir, exist_ok=True)
    return david_dir

def download_model(model_key: str) -> str:
    """Download a DAViD model if it doesn't exist."""
    if model_key not in MODEL_URLS:
        raise ValueError(f"download_model: Unknown model key: {model_key}")
    
    url = MODEL_URLS[model_key]
    filename = f"{model_key}.onnx"
    david_dir = get_david_models_path()
    model_path = os.path.join(david_dir, filename)
    
    if os.path.exists(model_path):
        logging.info(f"download_model: Model {model_key} already exists at {model_path}")
        return model_path
    
    logging.info(f"download_model: Downloading {model_key} from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"download_model: Downloading {model_key}: {percent:.1f}%")
        
        logging.info(f"download_model: Successfully downloaded {model_key} to {model_path}")
        return model_path
        
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)
        raise RuntimeError(f"download_model: Failed to download {model_key}: {str(e)}")

def numpy_to_comfy_image(np_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to ComfyUI IMAGE tensor (BHWC format).
    
    Args:
        np_array: Numpy array of shape (H, W), (H, W, C), or (B, H, W, C)
        
    Returns:
        torch.Tensor: ComfyUI IMAGE tensor of shape (B, H, W, C)
    """
    if np_array.ndim == 2:
        # Grayscale to RGB: (H, W) -> (H, W, 3)
        np_array = np.stack([np_array] * 3, axis=-1)
    elif np_array.ndim == 3:
        if np_array.shape[2] == 1:
            # Single channel to RGB: (H, W, 1) -> (H, W, 3)
            np_array = np.repeat(np_array, 3, axis=2)
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        np_array = np.expand_dims(np_array, axis=0)
    elif np_array.ndim == 4:
        # Already batched: (B, H, W, C)
        if np_array.shape[3] == 1:
            # Single channel to RGB: (B, H, W, 1) -> (B, H, W, 3)
            np_array = np.repeat(np_array, 3, axis=3)
    else:
        raise ValueError(f"numpy_to_comfy_image: Unsupported array shape: {np_array.shape}")
    
    # Ensure float32 and normalize to [0, 1]
    if np_array.dtype != np.float32:
        if np_array.dtype == np.uint8:
            np_array = np_array.astype(np.float32) / 255.0
        else:
            np_array = np_array.astype(np.float32)
    
    # Clamp to [0, 1] range
    np_array = np.clip(np_array, 0.0, 1.0)
    
    # Convert to tensor
    tensor = torch.from_numpy(np_array)
    return tensor

def numpy_to_comfy_mask(np_array: np.ndarray) -> torch.Tensor:
    """Convert numpy array to ComfyUI MASK tensor (BHW format).
    
    Args:
        np_array: Numpy array of shape (H, W), (H, W, C), or (B, H, W) or (B, H, W, C)
        
    Returns:
        torch.Tensor: ComfyUI MASK tensor of shape (B, H, W)
    """
    if np_array.ndim == 2:
        # Add batch dimension: (H, W) -> (1, H, W)
        np_array = np.expand_dims(np_array, axis=0)
    elif np_array.ndim == 3:
        if np_array.shape[2] > 1:
            # Multi-channel, take first channel: (H, W, C) -> (H, W)
            np_array = np_array[:, :, 0]
            # Add batch dimension: (H, W) -> (1, H, W)
            np_array = np.expand_dims(np_array, axis=0)
        else:
            # Single channel: (H, W, 1) -> (1, H, W)
            np_array = np_array[:, :, 0]
            np_array = np.expand_dims(np_array, axis=0)
    elif np_array.ndim == 4:
        # Batched with channels: (B, H, W, C) -> (B, H, W)
        np_array = np_array[:, :, :, 0]
    else:
        raise ValueError(f"numpy_to_comfy_mask: Unsupported array shape: {np_array.shape}")
    
    # Ensure float32 and normalize to [0, 1]
    if np_array.dtype != np.float32:
        if np_array.dtype == np.uint8:
            np_array = np_array.astype(np.float32) / 255.0
        else:
            np_array = np_array.astype(np.float32)
    
    # Clamp to [0, 1] range
    np_array = np.clip(np_array, 0.0, 1.0)
    
    # Convert to tensor
    tensor = torch.from_numpy(np_array)
    return tensor

def comfy_image_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert ComfyUI IMAGE tensor to numpy array.
    
    Args:
        tensor: ComfyUI IMAGE tensor of shape (B, H, W, C)
        
    Returns:
        np.ndarray: Numpy array of shape (B, H, W, C) for batch or (H, W, C) for single image
    """
    # Convert to numpy
    np_array = tensor.cpu().numpy()
    
    # Ensure values are in [0, 1] range
    np_array = np.clip(np_array, 0.0, 1.0)
    
    # Convert to uint8 range [0, 255] for processing
    np_array = (np_array * 255).astype(np.uint8)
    
    # If batch size is 1, squeeze the batch dimension for single image processing
    if np_array.shape[0] == 1:
        np_array = np_array[0]
    
    return np_array 