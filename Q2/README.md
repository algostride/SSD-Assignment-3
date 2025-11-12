# README — Box Blur Image Blurring Analysis

This repository contains the Jupyter Notebook `box_blur.ipynb`, which implements and analyzes two methods for applying a 3×3 mean blur (box blur) filter to a generated grayscale image: (1) an Iterative Python implementation using nested loops, and (2) a Vectorized NumPy implementation using array slicing and broadcasting. The notebook demonstrates the performance difference between both approaches and provides visual and analytical comparison.

## Requirements and Setup
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required Python libraries: NumPy, Pillow (PIL), Matplotlib  
Install dependencies with:  
`pip install numpy pillow matplotlib`

## Assumptions and Notes
- The image is programmatically generated — no external files are used.  
- The generated image is 400×200 pixels in grayscale mode ('L').  
- The text “SSD TAs ARE THE BEST” is dynamically scaled to fit the image boundaries.  
- The notebook attempts to use the font DejaVuSans-Bold.ttf; if unavailable, it defaults to a basic Pillow font.  
- The 3×3 mean blur replaces each pixel with the mean of its 8 neighbors and itself.  
- The outermost one-pixel border is excluded from the blur operation to simplify edge handling.  
- Benchmarking is performed using `time.perf_counter()` for accurate timing; results vary by hardware, but the NumPy implementation should consistently outperform the iterative Python version.  
- All visualizations are displayed inline using Matplotlib.  
- The notebook is intended for CPU-based execution; no GPU or external image processing libraries (like OpenCV) are required.

## How to Run
1. Open the notebook in Jupyter:
   `jupyter notebook box_blur.ipynb`
2. Run all cells sequentially to:
   - Generate the test image
   - Apply both blurring methods
   - Benchmark their performance
   - Display the resulting images and timing data

## Expected Outcome
You will see three images side by side: the original, the blurred result using the iterative method, and the blurred result using the NumPy vectorized method. The performance output should show the NumPy implementation running tens of times faster than the Python version.

## Summary
This notebook demonstrates how vectorization with NumPy significantly improves performance for pixel-wise operations due to compiled execution, contiguous memory layout, and reduced interpreter overhead. The project provides a clear example of why vectorized computation is preferred for large-scale numerical or image processing tasks.

