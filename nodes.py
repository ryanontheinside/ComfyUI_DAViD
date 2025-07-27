import os
import torch
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Import DAViD estimators
from .DAViD.runtime.multi_task_estimator import MultiTaskEstimator
from .DAViD.runtime.depth_estimator import RelativeDepthEstimator  
from .DAViD.runtime.soft_foreground_segmenter import SoftForegroundSegmenter
from .DAViD.runtime.surface_normal_estimator import SurfaceNormalEstimator

from .davide_utils import download_model, numpy_to_comfy_image, numpy_to_comfy_mask, comfy_image_to_numpy

class DAViDModelLoader:
    """Node to load DAViD models with automatic downloading."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["multitask_large", "depth_base", "depth_large", "foreground_base", "foreground_large", "normal_base", "normal_large"], {"default": "multitask_large"}),
            }
        }
    
    RETURN_TYPES = ("DAVID_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "DAViD"
    TITLE = "DAViD Model Loader"

    def load_model(self, model_type: str):
        """Load the specified DAViD model."""
        try:
            # Download model if needed
            model_path = download_model(model_type)
            
            # Create appropriate estimator based on model type
            if model_type == "multitask_large":
                model = MultiTaskEstimator(model_path)
            elif model_type.startswith("depth_"):
                model = RelativeDepthEstimator(model_path)
            elif model_type.startswith("foreground_"):
                model = SoftForegroundSegmenter(model_path)
            elif model_type.startswith("normal_"):
                model = SurfaceNormalEstimator(model_path)
            else:
                raise ValueError(f"load_model: Unknown model type: {model_type}")
            
            # Store model type for inference nodes
            model._david_model_type = model_type
            logging.info(f"load_model: Successfully loaded {model_type}")
            
            return (model,)
            
        except Exception as e:
            raise RuntimeError(f"load_model: Failed to load model {model_type}: {str(e)}")

class BaseDAViDInference:
    """Base class for DAViD inference nodes with shared functionality."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("DAVID_MODEL",),
            }
        }
    
    CATEGORY = "DAViD"
    
    def _prepare_batch(self, image_tensor: torch.Tensor) -> list:
        """Convert ComfyUI image tensor to list of numpy arrays for batch processing."""
        # Convert to numpy maintaining batch dimension
        np_batch = comfy_image_to_numpy(image_tensor)
        
        # If single image, wrap in list
        if np_batch.ndim == 3:
            return [np_batch]
        
        # If batch, convert to list of images
        return [np_batch[i] for i in range(np_batch.shape[0])]
    
    def _process_batch_results(self, results_list: list, result_type: str) -> torch.Tensor:
        """Combine batch results into a single tensor."""
        if len(results_list) == 1:
            # Single result - handle directly without stacking to avoid ambiguity
            single_result = results_list[0]
            
            if result_type == "mask":
                # Add batch dimension: (H, W) -> (1, H, W)
                batched = np.expand_dims(single_result, axis=0)
                tensor = torch.from_numpy(batched.astype(np.float32))
                tensor = torch.clamp(tensor, 0.0, 1.0)
                return tensor
            else:
                # Handle depth (H, W) and normal (H, W, C) cases
                if single_result.ndim == 2:
                    # Depth: normalize to [0,1] (closer = brighter)
                    depth = single_result.copy()
                    min_val, max_val = np.min(depth), np.max(depth)
                    if max_val != min_val:
                        depth_normalized = (depth - min_val) / (max_val - min_val)
                    else:
                        depth_normalized = np.ones_like(depth) * 0.5
                    
                    # Convert to RGB: (H, W) -> (H, W, 3) -> (1, H, W, 3)
                    rgb_result = np.stack([depth_normalized] * 3, axis=-1)
                    batched = np.expand_dims(rgb_result, axis=0)
                else:
                    # Normal: (H, W, C) -> (1, H, W, C) 
                    batched = np.expand_dims(single_result, axis=0)
                
                tensor = torch.from_numpy(batched.astype(np.float32))
                tensor = torch.clamp(tensor, 0.0, 1.0)
                return tensor
        else:
            # Multiple results - stack them
            if result_type == "mask":
                # Stack mask results: list of (H, W) -> (B, H, W)
                stacked = np.stack(results_list, axis=0)
                tensor = torch.from_numpy(stacked.astype(np.float32))
                tensor = torch.clamp(tensor, 0.0, 1.0)
                return tensor
            else:
                # Handle depth and normal stacking
                if results_list[0].ndim == 2:
                    # Depth results: normalize each (H, W) then convert to (H, W, 3) then stack
                    rgb_results = []
                    for depth in results_list:
                        # Normalize each depth map (closer = brighter)
                        min_val, max_val = np.min(depth), np.max(depth)
                        if max_val != min_val:
                            depth_normalized = (depth - min_val) / (max_val - min_val)
                        else:
                            depth_normalized = np.ones_like(depth) * 0.5
                        rgb_depth = np.stack([depth_normalized] * 3, axis=-1)
                        rgb_results.append(rgb_depth)
                    
                    stacked = np.stack(rgb_results, axis=0)
                else:
                    # Normal results: stack (H, W, C) -> (B, H, W, C)
                    stacked = np.stack(results_list, axis=0)
                
                tensor = torch.from_numpy(stacked.astype(np.float32))
                tensor = torch.clamp(tensor, 0.0, 1.0)
                return tensor

class DAViDMultiTaskEstimator(BaseDAViDInference):
    """Multi-task inference node for all DAViD tasks simultaneously."""
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("depth_map", "foreground_mask", "normal_map")
    FUNCTION = "estimate_all"
    TITLE = "DAViD Multi-Task Estimator"

    def estimate_all(self, image: torch.Tensor, model: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform multi-task estimation."""
        if not hasattr(model, 'estimate_all_tasks'):
            raise ValueError("estimate_all: Model must be a multi-task model")
        
        # Prepare batch
        image_batch = self._prepare_batch(image)
        
        # Process each image in batch
        depth_results = []
        foreground_results = []
        normal_results = []
        
        for np_image in image_batch:
            # Run inference
            results = model.estimate_all_tasks(np_image)
            depth_results.append(results["depth"])
            foreground_results.append(results["foreground"])
            normal_results.append(results["normal"])
        
        # Convert batch results to ComfyUI format
        depth_map = self._process_batch_results(depth_results, "image")
        foreground_mask = self._process_batch_results(foreground_results, "mask")
        normal_map = self._process_batch_results(normal_results, "image")
        
        return (depth_map, foreground_mask, normal_map)

class DAViDDepthEstimator(BaseDAViDInference):
    """Depth estimation inference node."""
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "estimate_depth"
    TITLE = "DAViD Depth Estimator"

    def estimate_depth(self, image: torch.Tensor, model: Any) -> Tuple[torch.Tensor]: 
        """Perform depth estimation."""
        # Prepare batch
        image_batch = self._prepare_batch(image)
        
        # Process each image in batch
        depth_results = []
        
        for np_image in image_batch:
            # Run inference - support both dedicated and multi-task models
            if hasattr(model, 'estimate_relative_depth'):
                depth_result = model.estimate_relative_depth(np_image)
            elif hasattr(model, 'estimate_all_tasks'):
                results = model.estimate_all_tasks(np_image)
                depth_result = results["depth"]
            else:
                raise ValueError("estimate_depth: Model does not support depth estimation")
            
            depth_results.append(depth_result)
        
        # Convert batch results to ComfyUI format
        depth_map = self._process_batch_results(depth_results, "image")
        
        return (depth_map,)

class DAViDForegroundSegmenter(BaseDAViDInference):
    """Foreground segmentation inference node."""
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("foreground_mask",)
    FUNCTION = "estimate_foreground"
    TITLE = "DAViD Foreground Segmenter"

    def estimate_foreground(self, image: torch.Tensor, model: Any) -> Tuple[torch.Tensor]:
        """Perform foreground segmentation."""
        # Prepare batch
        image_batch = self._prepare_batch(image)
        
        # Process each image in batch
        foreground_results = []
        
        for np_image in image_batch:
            # Run inference - support both dedicated and multi-task models
            if hasattr(model, 'estimate_foreground_segmentation'):
                foreground_result = model.estimate_foreground_segmentation(np_image)
            elif hasattr(model, 'estimate_all_tasks'):
                results = model.estimate_all_tasks(np_image)
                foreground_result = results["foreground"]
            else:
                raise ValueError("estimate_foreground: Model does not support foreground segmentation")
            
            foreground_results.append(foreground_result)
        
        # Convert batch results to ComfyUI format
        foreground_mask = self._process_batch_results(foreground_results, "mask")
        
        return (foreground_mask,)

class DAViDNormalEstimator(BaseDAViDInference):
    """Surface normal estimation inference node."""
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal_map",)
    FUNCTION = "estimate_normals"
    TITLE = "DAViD Normal Estimator"

    def estimate_normals(self, image: torch.Tensor, model: Any) -> Tuple[torch.Tensor]:
        """Perform surface normal estimation."""
        # Prepare batch
        image_batch = self._prepare_batch(image)
        
        # Process each image in batch
        normal_results = []
        
        for np_image in image_batch:
            # Run inference - support both dedicated and multi-task models
            if hasattr(model, 'estimate_normal'):
                normal_result = model.estimate_normal(np_image)
            elif hasattr(model, 'estimate_all_tasks'):
                results = model.estimate_all_tasks(np_image)
                normal_result = results["normal"]
            else:
                raise ValueError("estimate_normals: Model does not support normal estimation")
            
            normal_results.append(normal_result)
        
        # Convert batch results to ComfyUI format
        normal_map = self._process_batch_results(normal_results, "image")
        
        return (normal_map,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DAViDModelLoader": DAViDModelLoader,
    "DAViDMultiTaskEstimator": DAViDMultiTaskEstimator,
    "DAViDDepthEstimator": DAViDDepthEstimator,
    "DAViDForegroundSegmenter": DAViDForegroundSegmenter,
    "DAViDNormalEstimator": DAViDNormalEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DAViDModelLoader": "DAViD Model Loader",
    "DAViDMultiTaskEstimator": "DAViD Multi-Task Estimator", 
    "DAViDDepthEstimator": "DAViD Depth Estimator",
    "DAViDForegroundSegmenter": "DAViD Foreground Segmenter",
    "DAViDNormalEstimator": "DAViD Normal Estimator",
} 