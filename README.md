# PCB Defect Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=flat-square&logo=yolo&logoColor=white)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ttkbootstrap](https://img.shields.io/badge/GUI-ttkbootstrap-FF6B6B?style=flat-square&logo=python&logoColor=white)](https://ttkbootstrap.readthedocs.io)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/akhatova/pcb-defects)

## Description

Automated detection and classification system for Printed Circuit Board (PCB) defects using YOLO11. The project combines Kaggle training with a local graphical interface for testing trained models.

**Main Features:**
- Automatic detection of 6 types of PCB defects
- Modern GUI interface with ttkbootstrap theme
- Fast inference with PyTorch (GPU accelerated)
- Detection precision of 97.4%
- Optimized training pipeline (50% faster)

### GUI Interface Demo

![GUI Demo](results/demo.png)

Modern interface with ttkbootstrap theme featuring:
- Control panel with model and image loading
- Large image display area with zoom controls
- Color-coded detection results panel
- Real-time defect detection and visualization

## Technologies Used

| Component | Technology | Version | Usage |
|-----------|-------------|---------|-------|
| **Deep Learning** | YOLO11 (Ultralytics) | 8.3.0+ | Object detection |
| **Framework** | PyTorch | 2.2+ | Model training and inference |
| **GUI Interface** | ttkbootstrap | 1.10+ | Modern user interface |
| **Image Processing** | OpenCV, Pillow | 4.9+, 10.2+ | Image manipulation |
| **Training Platform** | Kaggle | - | Free GPU T4 |

## Model Optimizations

This project includes several optimizations for faster training without sacrificing performance:

- **50% Faster Training:** Reduced from 100 to 50 epochs (model converges at epoch 20)
- **Smart Early Stopping:** 15-epoch patience for automatic termination
- **Optimized Augmentations:** Balanced preprocessing speed and data diversity
- **Enhanced Bbox Precision:** Improved loss weights for better localization
- **Efficient Warmup:** Faster stabilization in first 3 epochs

## Performance Results

**Dataset:** [PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **Total Images:** 1,386 annotated images
- **Split:** 80% training / 10% validation / 10% test
- **Model:** YOLO11m (20M parameters)
- **Training:** 50 epochs on GPU T4 (optimized from 100 epochs)
- **Training Time:** Approximately 1-1.5 hours (50% faster than before)

### Global Metrics
| Metric | Score | Description |
|----------|-------|-------------|
| **Detection Precision** | **97.4%** | Mean Average Precision at IoU 0.5 |
| **Mean Precision** | **96.6%** | Precision across all classes |
| **Mean Recall** | **95.8%** | Detection rate |
| **F1-Score** | **96.2%** | Precision-recall balance |

### Performance by Class
| Class | Precision | Recall | Description |
|--------|-----------|--------|-------------|
| `missing_hole` | 99.5% | 100% | Missing drill hole |
| `open_circuit` | 99.4% | 100% | Broken trace |
| `mouse_bite` | 98.1% | 92.9% | Irregular edge |
| `short` | 96.4% | 94.1% | Short circuit |
| `spur` | 95.3% | 94.5% | Copper protrusion |
| `spurious_copper` | 95.8% | 93.1% | Unwanted copper |

### Training Visualization

![Training Results](results/training_results.png)

**Key Observations:**
- Rapid convergence in first 20 epochs
- Stable training with no overfitting
- Validation errors near zero after epoch 10
- Learning rate smoothly decays from 0.01 to 0.0001

**Optimizations Applied:**
- Training reduced to 50 epochs (model converges at epoch 20)
- Early stopping with 15-epoch patience
- Faster warmup (3 epochs instead of 5)
- Optimized augmentations for faster preprocessing
- Improved bounding box precision with enhanced loss weights

**Result:** 50% faster training time with maintained or improved performance.

### Sample Detection Results

![Sample Predictions](results/sample_predictions.png)

The model successfully detects various PCB defects with high confidence and accurate bounding boxes across different defect types.

## Defect Classes

| ID | Class | Description |
|----|--------|-------------|
| 0 | `missing_hole` | Missing drill hole |
| 1 | `mouse_bite` | Irregular edge |
| 2 | `open_circuit` | Broken trace |
| 3 | `short` | Short circuit |
| 4 | `spur` | Copper protrusion |
| 5 | `spurious_copper` | Unwanted copper |

## Training on Kaggle

1. **Create Kaggle notebook** with GPU enabled
2. **Add dataset:** `akhatova/pcb-defects`
3. **Run training code:**

```python
!pip install ultralytics -q
!wget -q https://github.com/alainpaluku/pcb-defect-detection/archive/main.zip
!unzip -q main.zip
!mv pcb-defect-detection-main pcb-defect-detector
%cd pcb-defect-detector
!python run_kaggle.py
```

4. **Download trained model:**

```python
from IPython.display import FileLink
FileLink('/kaggle/working/pcb_model.pt')
```

## Local GUI Testing

```bash
git clone https://github.com/alainpaluku/pcb-defect-detection.git
cd pcb-defect-detection
pip install -r requirements.txt
```

1. **Place downloaded model** in `models/` directory:
   - `pcb_model.pt`

2. **Place test images** in `images/` directory

3. **Run GUI:** `python -m gui_test.app`

## Project Architecture

```
pcb-defect-detection/
├── src/                   # Main Python modules
│   ├── model.py          # YOLO11 model and export
│   ├── trainer.py        # Training pipeline
│   ├── detector.py       # Detection interface
│   └── config.py         # Configuration
├── gui_test/             # Graphical interface
│   ├── app.py           # Main application
│   ├── main_window.py   # Main window
│   └── model_loader.py  # Model loading
├── models/               # Trained models (.pt)
├── images/               # Test images
├── run_kaggle.py         # Kaggle training script
└── requirements.txt      # Dependencies

```

---

**Author:** Alain Paluku - [@alainpaluku](https://github.com/alainpaluku)