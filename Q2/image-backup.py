# Implementation and Performance Analysis of Image Blurring (Fixed Text Bounds)

import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# --- 1. Generate input image (Fixed text fit) -------------------------------
W, H = 400, 200
text = "SSD TAs ARE THE BEST"

# Create grayscale image with black background
img_pil = Image.new('L', (W, H), 0)
draw = ImageDraw.Draw(img_pil)

# Try to load a TrueType font and dynamically adjust size to fit width
font_size = 60
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
except Exception:
    font = ImageFont.load_default()

# Reduce font size until text fits inside image boundaries
while True:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    if text_w <= W - 10 and text_h <= H - 10:
        break
    font_size -= 2
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        break

# Center the text
pos = ((W - text_w) // 2, (H - text_h) // 2)

# Draw white text
draw.text(pos, text, fill=255, font=font)

# Convert to NumPy array
original_image = np.array(img_pil, dtype=np.uint8)

# --- 2. Iterative blur implementation ---------------------------------------
def blur_python(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    out = image.copy()
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            region = image[y - 1:y + 2, x - 1:x + 2]
            out[y, x] = np.mean(region, dtype=np.float32)
    return out

# --- 3. Vectorized NumPy blur implementation -------------------------------
def blur_numpy(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    p = np.pad(image, 1, mode='constant', constant_values=0).astype(np.float32)
    s = (
        p[0:h, 0:w] + p[0:h, 1:w+1] + p[0:h, 2:w+2]
        + p[1:h+1, 0:w] + p[1:h+1, 1:w+1] + p[1:h+1, 2:w+2]
        + p[2:h+2, 0:w] + p[2:h+2, 1:w+1] + p[2:h+2, 2:w+2]
    )
    out = image.copy()
    out[1:-1, 1:-1] = (s[1:-1, 1:-1] / 9).astype(np.uint8)
    return out

# --- 4. Benchmarking -------------------------------------------------------
def benchmark(func, img, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = func(img)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), sum(times)/len(times)

min_py, avg_py = benchmark(blur_python, original_image)
min_np, avg_np = benchmark(blur_numpy, original_image)

print(f"blur_python: {avg_py:.6f}s, blur_numpy: {avg_np:.6f}s, speedup: {avg_py/avg_np:.2f}x")

# --- 5. Visualization ------------------------------------------------------
blurred_py = blur_python(original_image)
blurred_np = blur_numpy(original_image)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original Image')
axes[1].imshow(blurred_py, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Blurred (Iterative Python)')
axes[2].imshow(blurred_np, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Blurred (Vectorized NumPy)')
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()

# --- 6. Analysis -----------------------------------------------------------
analysis = """
Analysis and Discussion
-----------------------
1) Performance Results:
   - The vectorized NumPy implementation (blur_numpy) is significantly faster than the
     iterative Python implementation (blur_python). The exact speedup will vary by
     machine, but on typical hardware you should expect a multi-fold speedup (often
     tens of times faster) because blur_numpy leverages compiled C loops inside NumPy.

2) Core Concepts:
   a) Vectorization:
      - In blur_numpy we avoid Python-level loops and instead operate on whole arrays
        and array slices. NumPy performs the heavy lifting in optimized C code, allowing
        the operation to run much faster for large arrays.

   b) Compiled Code vs. Interpreter:
      - NumPy's internal operations (slicing, arithmetic on arrays) are implemented in
        compiled C. This means the expensive per-element work happens outside the
        Python interpreter. In contrast, blur_python runs Python-level loops where each
        iteration incurs interpreter overhead (bytecode execution, index checks), which
        is much slower for large arrays.

   c) Memory Layout:
      - NumPy stores arrays in contiguous blocks of memory (row-major by default). This
        contiguous layout enables efficient memory access patterns (better cache locality)
        when NumPy performs vectorized operations. Iterative Python loops tend to cause
        many small, scattered memory accesses and repeated Python overhead, reducing
        performance.

Conclusion: For numerical image processing tasks where the same operation is applied
across many pixels, a vectorized implementation using NumPy is both simpler to write
(and often more readable) and substantially faster due to lower interpreter overhead
and optimized, cache-friendly compiled code paths.
"""

print(analysis)