# NNDL Project

Dual-head image classifier that predicts both a superclass and a subclass. The pipeline builds dataloaders from CSV metadata, trains a transfer-learning backbone with two classification heads, calibrates open-set thresholds using near- and far-OOD data, and reports closed- and open-set metrics.

## Repository layout
- data/: zipped raw images, label metadata, and prepared splits (train/val/calibration)
- scripts/: helper utilities for split creation, OOD set generation, and visualizations
- src/: model, data, training, inference, and visualization code
- checkpoints/: training outputs (created when you run training)
- figures/ and experiments/: saved plots and example results

## Setup
1) Use Python 3.10+ and create a virtual environment
2) Install dependencies:
```
pip install -r requirements.txt
```
   This also pulls the OpenAI CLIP package from GitHub

## Data preparation
1) Unzip the raw image archives into the expected folders:

```
unzip data/raw/train_images.zip -d data/raw/train_images
unzip data/raw/test_images.zip  -d data/raw/test_images
```

Metadata lives in `data/meta/`:
  - `train_data.csv` columns: image, superclass_index, subclass_index, description
  - `superclass_mapping.csv` and `subclass_mapping.csv` map indices to class names

2) Dataset splits are pre-generated under `data/splits/`

To regenerate them with the same stratification logic, run:

```
python scripts/create_splits.py
```

3) Far-OOD calibration images: writes JPEGs to `data/raw/farood_images` 

```
python scripts/create_far_ood.py
```

4) Near-OOD examples are created on-the-fly from the calibration split with strong corruptions

## Training & Evaluation 
Train a dual-head model and optionally fine-tune the backbone:

```
python -m src.training.main \
  --mode clip_b32 \ 
  --train_csv data/splits/train_split.csv \
  --val_csv data/splits/val_split.csv \
  --calib_csv data/splits/calibration_split.csv \
  --images_root data/raw/train_images \
  --far_ood_root data/raw/farood_images \
  --superclass_map data/meta/superclass_mapping.csv \
  --subclass_map data/meta/subclass_mapping.csv \
  --batch_size 64 \
  --num_epochs 15 \
  --epochs_finetune 10 
```

- `--fine_tune/--no_fine_tune` toggles backbone unfreezing for the second stage
- Optimizers: `adam`, `adamw` (default), `sgd`
- Scheduler: `cosine` or `none`
- Outputs:
  - model checkpoints saved under `checkpoints/<mode>/` (`best_head.pth`, `best_finetune.pth`)
  - closed-set and open-set metrics written to `metric_output.csv`

## Notes
- GPU is recommended but the code will fall back to CPU if CUDA is unavailable
