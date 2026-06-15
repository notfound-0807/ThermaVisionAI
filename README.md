# 🌡️ ThermaVision AI

## 🌐 Live Demo
https://thermavisionai-msjqestynu7tpc3qzy3nut.streamlit.app/

ThermaVision AI is a Streamlit web application that processes thermal and infrared images through a five-stage pipeline:

1. **Grayscale Conversion** — Normalizes multi-channel inputs to single-channel intensity maps
2. **Non-Local Means Denoising** — Removes thermal sensor noise while preserving structural edges
3. **CLAHE** — Adaptive local contrast enhancement to reveal detail in both hot and cold regions
4. **Edge Sharpening** — Laplacian kernel to accentuate object boundaries
5. **Pseudo-Color Mapping** — False-color colormaps (Inferno, Jet, Plasma, Turbo) for perceptual interpretation

---

## Project Structure

```
ThermaVisionAI/
├── app.py            # Main Streamlit application
├── enhancement.py    # Image enhancement pipeline (denoise, CLAHE, sharpen)
├── colorization.py   # Pseudo-color thermal colorization module
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Quick Start

### 1. Clone / copy the project
```bash
cd ThermaVisionAI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Features

| Feature | Description |
|---|---|
| Image Upload | JPG, JPEG, PNG — grayscale or color IR images |
| Enhancement Pipeline | 4-stage processing with intermediate previews |
| Colorization | 4 OpenCV colormaps with live switching |
| Comparison Dashboard | Side-by-side original / enhanced / colorized |
| Quality Metrics | Mean intensity, contrast improvement %, resolution |
| Download | Enhanced + colorized outputs as PNG |

---

## Colormap Guide

| Colormap | Best For |
|---|---|
| **Inferno** | Scientific reporting; perceptually uniform, accessible |
| **Jet** | Classic thermal visualization; widely recognized |
| **Plasma** | Subtle temperature gradients; smooth transitions |
| **Turbo** | Full-spectrum with better perceptual linearity than Jet |

---

## Technical Details

### Enhancement Module (`enhancement.py`)
- `fastNlMeansDenoising` — h=10, template 7×7, search 21×21
- `cv2.createCLAHE` — clipLimit=2.0, tileGrid=(8,8)
- Sharpening kernel: `[[0,-1,0],[-1,5,-1],[0,-1,0]]`

### Colorization Module (`colorization.py`)
- Applies `cv2.applyColorMap` with selected OpenCV constant
- Converts BGR→RGB for Streamlit display
- Supports runtime switching without re-running the enhancement pipeline

---

## Dependencies

- **Python** 3.9+
- **Streamlit** ≥ 1.32
- **OpenCV** ≥ 4.9 (headless build)
- **NumPy** ≥ 1.26
- **Pillow** ≥ 10.2

---

## Use Cases

- Satellite thermal band visualization
- UAV/drone night-vision enhancement
- Medical thermography preprocessing
- Industrial heat-map analysis
- Forest fire / hotspot detection

---
