# ComfyUI DAViD Custom Node

ComfyUI wrapper for [DAViD (Data-efficient and Accurate Vision Models from Synthetic Data)](https://microsoft.github.io/DAViD) - human-centric computer vision models for depth estimation, foreground segmentation, and surface normal estimation.

## Features

- **Automatic Model Downloads**: Models are automatically downloaded to `models/DAViD/` when first used
- **Multiple Model Options**: Support for both Base (ViT-B/16) and Large (ViT-L/16) variants
- **Efficient Multi-Task Processing**: Single model can perform all three tasks simultaneously
- **ComfyUI Native Outputs**: Proper IMAGE and MASK tensor formats for seamless workflow integration

## Available Nodes

### DAViD Model Loader
Loads and manages DAViD ONNX models with automatic downloading.

**Inputs:**
- `model_type`: Dropdown selection from available models
  - `multitask_large` (recommended) - Performs all tasks simultaneously
  - `depth_base` / `depth_large` - Depth estimation only
  - `foreground_base` / `foreground_large` - Foreground segmentation only  
  - `normal_base` / `normal_large` - Surface normal estimation only

**Outputs:**
- `model`: DAVID_MODEL type for use with inference nodes

### DAViD Multi-Task Estimator
Performs all three tasks simultaneously using the multi-task model (most efficient).

**Inputs:**
- `image`: IMAGE - Input image to process
- `model`: DAVID_MODEL - Multi-task model from loader

**Outputs:**
- `depth_map`: IMAGE - Relative depth map visualization
- `foreground_mask`: MASK - Human silhouette segmentation
- `normal_map`: IMAGE - Surface normal map visualization

### Individual Task Nodes

#### DAViD Depth Estimator
**Outputs:** `depth_map` (IMAGE)

#### DAViD Foreground Segmenter  
**Outputs:** `foreground_mask` (MASK)

#### DAViD Normal Estimator
**Outputs:** `normal_map` (IMAGE)

## Usage Example

1. Add **DAViD Model Loader** node, select `multitask_large`
2. Add **DAViD Multi-Task Estimator** node
3. Connect your input IMAGE to the estimator
4. Connect the model output from loader to estimator
5. Use the three outputs (depth, mask, normals) in your workflow

## Technical Details

- **Input Resolution**: Models expect 384x384 input (automatically handled)
- **Output Format**: 
  - IMAGE tensors: (1, H, W, 3) in range [0, 1]
  - MASK tensors: (1, H, W) in range [0, 1]
- **Model Storage**: Downloaded to `{ComfyUI}/models/DAViD/`
- **Performance**: Multi-task model is most efficient for getting all outputs

## Dependencies

Required packages (automatically installed with DAViD):
- numpy
- onnx  
- onnxruntime-gpu
- opencv-python
- torch (from ComfyUI)

## Model Information

All models are trained on synthetic human data and optimized for human-centric scenes. Models are provided under MIT License by Microsoft Research.

For more details, see the [DAViD paper](https://arxiv.org/abs/2507.15365) and [official repository](https://github.com/microsoft/DAViD). 