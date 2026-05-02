# YOLOv13-GF Prediction Package

Detect L-T dwarfs from Pan-STARRS 5-channel images 

## 📁 Structure

```
.
├── ultralytics/           # YOLO code
├── weights/best.pt        # Model
├── images/                # Input: 5-channel .npy files
├── predict/labels/        # Output: TXT annotations
├── data.yaml              # Class config
├── predict.ipynb          # Prediction script
├── annotate_channels.py   # Visualization module
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Prepare Images
- **Format**: `.npy` only (5-channel: g, r, i, z, y)
- **Source**: Pan-STARRS1 survey data
- **Shape**: `(5, H, W)` or `(H, W, 5)`

⚠️ **No jpg/png support**

### 3. Configure
Edit `predict.ipynb`:
```python
WEIGHTS_PATH = "./weights/best.pt"
ORIGINAL_IMAGES_PATH = "./images"
CONF_THRESHOLD = 0.778  # Adjust as needed
IOU_THRESHOLD = 0.45
BATCH_SIZE = 16
DEVICE = "cpu"  # or "cuda"
```

### 4. Run
```bash
jupyter notebook predict.ipynb
# Execute all cells
```

## 📊 Output

**TXT Files** (`predict/labels/`):
- Format: `<class_id> <x_center> <y_center> <width> <height> <confidence>`
- One file per input image

## 🔧 Optional: Channel Visualization

Use `annotate_channels.py` to visualize detections on each channel:

```python
from annotate_channels import batch_annotate_from_txt

results = batch_annotate_from_txt(
    npy_dir="./images",
    txt_dir="./predict/labels",
    class_names=['LT dwarf'],
    save_dir="./annotate_output",
    show_plot=True
)
```

Output: `annotate_output/*_5channels_annotated.jpg`

## ⚙️ Config (`data.yaml`)

```yaml
nc: 1
names: ['LT dwarf']
```

## ⚠️ Notes

1. **5-channel npy only** - No regular images
2. **Verify paths** in notebook before running
3. **Python 3.10** required for dependencies

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| No .npy found | Check `images/` directory |
| Model load failed | Verify `weights/best.pt` exists |
| Import error | Run `pip install -r requirements.txt` |
| OOM error | Reduce `BATCH_SIZE` |

## 📦 Dependencies

Core: `torch`, `ultralytics`, `astropy`, `opencv-python`, `numpy`, `matplotlib`  
Auto: `timm`, `huggingface_hub`, `einops`

See `requirements.txt` for versions.
