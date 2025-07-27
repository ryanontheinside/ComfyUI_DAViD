# ComfyUI_DAViD/__init__.py

from .nodes import (
    DAViDModelLoader,
    DAViDMultiTaskEstimator,
    DAViDDepthEstimator,
    DAViDForegroundSegmenter,
    DAViDNormalEstimator,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS
)

# Export the mappings for ComfyUI to discover
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]