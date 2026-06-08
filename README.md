# YOLOv13 Inference for Pan-STARRS L/T Dwarf Candidate Search

This repository provides the inference code used to apply a trained YOLOv13 detector to Pan-STARRS five-channel image cutouts for L/T dwarf candidate search.

## Why this repository exists

L/T dwarfs are faint in optical survey images, and catalog-based color selection can miss sources when catalog photometry is incomplete, low signal-to-noise, or unavailable in key bands. The manuscript uses image-level object detection as a complementary candidate-generation step: the detector searches directly in Pan-STARRS multi-band image cutouts and returns candidate source positions and confidence scores. The detected sources can then be checked with external photometry, photo-type classification, template fitting, proper-motion information, and follow-up observations.

This repository is provided so that readers can inspect and run the inference part of that workflow. It is intended to show how the trained detector is applied to five-channel Pan-STARRS `.npy` images and how the image-level detections are exported for later catalog matching and validation. It is not intended to reproduce the full training procedure, photo-type classification, LePhare fitting, proper-motion reassessment, or final catalog construction.

## Repository scope

This repository contains the wide-field inference package used after model training. It includes:

- the trained YOLOv13 model weight used for inference;
- example five-channel Pan-STARRS `.npy` image inputs;
- a notebook for running prediction on `.npy` cutouts;
- local inference code required for five-channel NumPy inputs;
- a visualization script for checking detections in each Pan-STARRS band;
- example YOLO-format output labels.

The repository is inference-only. The full training pipeline and the downstream astrophysical validation workflow are not included.

## Structure

```text
.
├── ultralytics/           # Local inference code used by this repository
├── weights/best.pt        # Trained YOLOv13 model weight
├── images/                # Input five-channel .npy files
├── predict/labels/        # Output YOLO-format TXT detections
├── data.yaml              # Class configuration
├── predict.ipynb          # Inference notebook
├── annotate_channels.py   # Visualization module for five-channel cutouts
└── requirements.txt       # Python dependencies
```

## Important note on the local inference code

This repository includes local inference code required for Pan-STARRS five-channel `.npy` inputs. Run the notebook from the repository root and do not replace the local `ultralytics/` directory with the standard package when reproducing this inference workflow.

The local code is included only to support inference with the data format used in this project. The public release is intended to make the large-scale prediction step inspectable and reusable.

## Installation

A Python 3.10 environment is recommended.

```bash
pip install -r requirements.txt
```

If your environment already contains another version of `ultralytics`, run the notebook from the repository root so that the local code in this repository is used.

## Input data

The inference notebook expects Pan-STARRS five-channel NumPy arrays.

- Format: `.npy` only
- Bands: `g, r, i, z, y`
- Supported shapes: `(5, H, W)` or `(H, W, 5)`
- Regular `.jpg` or `.png` images are not supported as model inputs

The example files in `images/` are provided to demonstrate the expected input format.

## Configuration

Edit the configuration block in `predict.ipynb` before running inference:

```python
WEIGHTS_PATH = "./weights/best.pt"
ORIGINAL_IMAGES_PATH = "./images"
DATA_YAML_PATH = "./data.yaml"

CONF_THRESHOLD = 0.902
IOU_THRESHOLD = 0.45
BATCH_SIZE = 16
DEVICE = "cpu"  # or "cuda"
```

The confidence threshold `0.902` corresponds to the threshold adopted for the wide-field inference described in the manuscript.

## Run inference

Start Jupyter and execute all cells in the notebook:

```bash
jupyter notebook predict.ipynb
```

The notebook loads the trained weight from `weights/best.pt`, reads five-channel `.npy` files from `images/`, runs YOLOv13 inference, and saves detections in YOLO TXT format.

## Output format

Prediction labels are saved in:

```text
predict/labels/
```

Each output file corresponds to one input `.npy` image. Each line follows the format:

```text
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

The coordinates are normalized YOLO-format image coordinates. They can be converted back to pixel coordinates and then, using the WCS information from the corresponding Pan-STARRS FITS image, to celestial coordinates for catalog matching.

## Optional channel visualization

Use `annotate_channels.py` to visualize detections on each Pan-STARRS channel:

```python
from annotate_channels import batch_annotate_from_txt

results = batch_annotate_from_txt(
    npy_dir="./images",
    txt_dir="./predict/labels",
    class_names=["LT dwarf"],
    save_dir="./annotate_output",
    show_plot=True
)
```

Annotated images are written to:

```text
annotate_output/
```

This visualization step is for inspection only. It is not used as model input.

## Class configuration

The detector uses one target class:

```yaml
nc: 1
names: ["LT dwarf"]
```

## Troubleshooting

| Issue | Possible solution |
|---|---|
| No `.npy` files are found | Check `ORIGINAL_IMAGES_PATH` and confirm that the input files are in `.npy` format. |
| Model loading fails | Confirm that `weights/best.pt` exists and that the path in `predict.ipynb` is correct. |
| Import errors occur | Install the dependencies with `pip install -r requirements.txt` and run from the repository root. |
| Standard `ultralytics` package is imported instead of local code | Run the notebook from the repository root and avoid replacing the local `ultralytics/` directory. |
| Out-of-memory error | Reduce `BATCH_SIZE` or use CPU inference for small tests. |
| Visualization fails | Check that the corresponding TXT files exist in `predict/labels/` and that `annotate_channels.py` is run with the correct paths. |

## Relationship to the manuscript

This repository corresponds to the inference stage of the manuscript. The model output from this step provides image-level candidate detections. The subsequent scientific validation described in the manuscript, including duplicate removal, cross-matching with external catalogs, photo-type classification, LePhare checks, proper-motion reassessment, and construction of the final candidate catalogs, is performed outside this inference-only repository.

## Code assistance

ChatGPT was used for code-assistance during preparation and refinement of parts of this repository, including code organization, debugging suggestions, and documentation support. All released code and outputs were reviewed and validated by the authors. The scientific analysis, candidate validation, and interpretation are the responsibility of the authors.

## License and archival release

This repository should be distributed with a reuse license compatible with the included local inference code. Because the repository includes local code derived from Ultralytics, the release should include an appropriate license file and retain the corresponding source code.

A formal tagged release of the repository should be created for the manuscript revision. The tagged release will be archived in a DOI-issuing repository such as Zenodo before final publication, and the manuscript will cite both the GitHub repository and the archived DOI release.
