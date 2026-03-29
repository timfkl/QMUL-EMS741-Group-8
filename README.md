# EMS741 Group 8 — Few-Shot Abdominal MRI Segmentation

A Reptile meta-learning approach to few-shot anatomical structure segmentation in 2D abdominal MR images, developed for the EMS741 Deep Learning module at Queen Mary University of London.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Running the Notebook](#running-the-notebook)
- [Dataset](#dataset)
- [Method](#method)
- [Results](#results)
- [Configuration](#configuration)
- [Code Structure](#code-structure)
- [Contributors](#contributors)

---

## Overview

Segmenting new anatomical structures in MRI scans typically requires large annotated datasets — which are expensive and time-consuming to produce. This project investigates **few-shot segmentation**: given only 1, 3, or 5 annotated examples of a new structure, can a model adapt quickly and segment it reliably?

We implement:
- **Reptile** (Nichol et al., 2018) — a first-order meta-learning algorithm that trains a shared model initialisation which can rapidly fine-tune to new tasks
- **Baseline** — a U-Net trained from scratch on the same few examples, with cosine LR scheduling

Both methods use the same compact U-Net architecture and are evaluated under identical few-shot conditions for a fair comparison.

---

## Repository Structure
.
├── EMS741_Group8_Reptile_Cluster.ipynb # Main notebook (run on Colab)
├── core_methods.py # All core logic: model, data, training
├── results/ # Output zips from each run
└── README.md


`core_methods.py` is hosted publicly on GitHub and downloaded automatically by the notebook at runtime — no manual upload needed on Colab.

---

## Setup

### Google Colab (recommended)

1. Open the notebook in Colab
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Click **Run All**

The notebook will automatically:
- Download `core_methods.py` from this repository
- Download and extract the dataset if not already present

No other setup is required.

### JupyterHub / Local

Set the dataset path before launching Jupyter:

```bash
export EMS741_DATA_ROOT=/path/to/dataset
jupyter notebook
```

If `EMS741_DATA_ROOT` is not set, the dataset will be downloaded to the current working directory.

Ensure `core_methods.py` is in the same directory as the notebook.

---

## Running the Notebook

The notebook is divided into clearly labelled sections:

| Section | Description |
|---------|-------------|
| 1. Dataset Setup | Auto-downloads dataset if not found |
| 2. Imports | Downloads `core_methods.py` from GitHub, imports all dependencies |
| 3. Reproducibility & Device | Set `BASE_SEED` and detect GPU |
| 4. Task Discovery | Scans dataset directories for train/val/test tasks |
| 5. Reptile Meta-Training | Trains the meta-model over `N_OUTER` outer steps |
| 6. Restore Best Checkpoint | Loads best weights saved during training |
| 7. Few-Shot Evaluation | Evaluates Reptile vs baseline at 1/3/5-shot |
| 8. Training Curves | Plots validation Dice and meta-LR decay |
| 9. Qualitative Visualisation | Side-by-side MR image / GT / predictions per task |
| 10. Results Plot | Error bar plot of mean ± std Dice across seeds |
| 11. Export Results | Bundles all outputs into a timestamped `.zip` |

**To reproduce our results**, set `BASE_SEED` to one of: `42`, `123`, `7` and run all cells. Our reported results aggregate across all three seeds.

**To load a previously saved checkpoint** instead of retraining, replace `metamodelbest_checkpoint` in Section 6 with the path to your `.pt` file.

---

## Dataset

The dataset consists of 2D MRI slices and binary segmentation masks for multiple abdominal anatomical structures. Each structure is a separate task.

**Expected directory structure:**
DATA_ROOT/
├── train/
│ ├── task1/
│ │ ├── images/ # *.png grayscale slices
│ │ └── masks/ # *.png binary segmentation masks
│ └── task2/ ...
├── val/
│ └── taskX/ ...
└── test/
└── taskY/ ...

Images are resized to `256×256` on load. Masks are binarised at threshold 127 if not already binary. The dataset is available at:
https://zenodo.org/records/18745413/files/ems741_cw_data.zip

---

## Method

### U-Net Architecture

A compact encoder-decoder U-Net with:
- **Input:** 1-channel greyscale MRI slice
- **Output:** 1-channel sigmoid probability mask
- **Encoder:** 4 DoubleConv blocks with MaxPool downsampling (32→64→128→256 channels)
- **Decoder:** Transposed convolution upsampling with skip connections
- **Normalisation:** GroupNorm (8 groups) by default — more stable than BatchNorm at small batch sizes

### Reptile Meta-Training

Reptile learns a weight initialisation that can adapt quickly to new tasks:

1. Sample a random training task
2. Clone the meta-model and run `K_INNER` SGD steps on support examples
3. Update meta-weights toward the fine-tuned weights:

$$\theta \leftarrow \theta + \alpha \cdot (\tilde{\theta} - \theta)$$

where $\alpha$ is the (linearly decaying) meta learning rate and $\tilde{\theta}$ are the inner-loop weights.

Validation is run every `VAL_EVERY` steps using 5-shot adaptation on the val split. The best checkpoint by val Dice is saved automatically.

### Loss Function

Combined BCE + Dice loss:

$$\mathcal{L} = 0.2 \cdot \mathcal{L}_{\text{BCE}} + 0.8 \cdot \mathcal{L}_{\text{Dice}}$$

### Baseline

A U-Net initialised from scratch and fine-tuned on the same `n_shot` support examples using Adam + cosine LR annealing — identical inner loop to Reptile adaptation for a fair comparison.

---

## Results

Aggregated across 3 random seeds (42, 123, 7), 3 episodes each:

| Shots | Reptile (mean ± std) | Baseline (mean ± std) |
|-------|---------------------|----------------------|
| 1-shot | **0.192 ± 0.038** | 0.130 ± 0.018 |
| 3-shot | **0.209 ± 0.024** | 0.152 ± 0.013 |
| 5-shot | **0.182 ± 0.023** | 0.137 ± 0.037 |

Reptile consistently outperforms the from-scratch baseline across all shot counts. Best single-run val Dice: **0.214** (seed 42, step 7600).

---

## Configuration

Key hyperparameters are set at the top of the training section in the notebook. Defaults used in our experiments:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `N_OUTER` | 8000 | Total outer meta-training steps |
| `K_INNER` | 10 | Inner SGD steps per task |
| `INNER_LR` | 0.003 | Inner loop learning rate |
| `META_LR` | 0.01 | Initial meta learning rate (linearly decayed) |
| `ADAPT_STEPS` | 30 | Fine-tuning steps at test time |
| `BASELINE_EPOCHS` | 30 | Baseline training steps |
| `BASE_SEED` | 42 / 123 / 7 | Random seed |

All other defaults are in `DEFAULT_CONFIG` inside `core_methods.py`.

---

## Code Structure

### `core_methods.py`

| Component | Description |
|-----------|-------------|
| `UNet` | Compact encoder-decoder segmentation model |
| `DoubleConv` | Two Conv→Norm→ReLU blocks, used in UNet |
| `SegDataset` | PyTorch Dataset with optional augmentation (hflip, vflip, rotate) |
| `FewShotEpisodeDataset` | Deterministic support/query split given a seed |
| `discover_tasks` | Scans a split directory and returns task path dicts |
| `load_sample` | Loads and resizes a single image-mask pair |
| `bce_dice_loss` | Combined BCE + Dice loss (0.2/0.8 weighting) |
| `dice_score` | Binary Dice metric (threshold 0.2) |
| `run_inner_loop` | Shared gradient loop used by all training paths |
| `reptile_meta_train` | Full Reptile outer loop with val, checkpointing, LR decay |
| `adapt_and_evaluate` | Fine-tunes meta-model on support set, evaluates on query |
| `unified_adapt_and_evaluate` | Generic wrapper used for baseline evaluation |
| `evaluate_few_shot` | Averages `adapt_and_evaluate` across all tasks in a split |
| `train_baseline` | *(Deprecated)* Standalone baseline trainer, superseded by `unified_adapt_and_evaluate` |

---

## Contributors

Group 8 — EMS741 Deep Learning for Data and Image Analysis, QMUL 2025/26