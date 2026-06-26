# TransformerDownscaling_MP: Transformer-Based Statistical Downscaling for Precipitation

Estimating high-resolution (1–4 km) daily precipitation over India using space/time deep learning. This repository implements a **2× statistical downscaling** framework transforming coarse-resolution precipitation (MSWEP 0.1°) to fine-resolution targets (CHIRPS 0.05°), conditioned on wind fields and topography.

## Overview


### Core Contribution

This work develops **ClimateUformerMultiscaleFuseModel**, a transformer-based architecture (fork of Zhong et al. 2024, QJRMS) that addresses the systematic dry bias in precipitation downscaling through:

- **Asymmetric loss function (AsymLoss v2)** with extreme-event weighting and under-prediction penalties
- **Multi-scale deep supervision** with pixel-shuffle upsampling
- **Topography fusion branch** encoding SRTM elevation as a conditioning signal
- **Windowed self-attention** (LeWin blocks, window 16×16) for computational efficiency

**Key Result:** Reduces systematic dry bias by ~23% and extends maximum predicted precipitation from ~66 mm/day (Charbonnier baseline) to ~204 mm/day.

---

## Data & Architecture

### Input Data

| Dataset | Resolution | Role | Coverage |
|---------|-----------|------|----------|
| **MSWEP** | 0.1° (~10 km) | Low-resolution precipitation input | Global daily 2015–2020 |
| **CHIRPS** | 0.05° (~5 km) | High-resolution target (labels) | India-focused 2015–2020 |
| **ERA5-Land** | 0.1° (~11 km) | Wind conditioning (u, v components) | Global 2015–2020 |
| **SRTM** (USGS SRTMGL1_003) | ~30 m | Elevation/topography branch | India coverage |

### Model Architecture

```
Input (LR): MSWEP 64×64 patches @ 0.1° + ERA5-Land winds
         ↓
    LeWin Transformer Blocks (windowed self-attention)
         ↓
    Multi-scale Deep Supervision + Pixel-Shuffle Upsampling
         ↓
    Topography Fusion Branch (SRTM conditioning)
         ↓
Output (HR): Downsampled CHIRPS 128×128 patches @ 0.05°
```

**Architecture Details:**
- Windowed self-attention window size: 16×16
- Upsampling: pixel-shuffle (2×)
- Loss: v2 AsymLoss (Charbonnier base + extreme-event weighting + asymmetric penalty + soft BCE occurrence term)
- Training data normalization: log₁ₚ + z-score (computed over land pixels only)
- Land-sea masking: applied at both LR (0.1°) and HR (0.05°) resolutions

---

## Setup & Installation

### Environment Requirements

- **Python:** 3.9–3.11
- **CUDA:** 11.8+ (recommended for GPU training)
- **PyTorch:** 2.0+
- **Training Platform:** Kaggle (2× GPU sessions, 30-hour kernel timeout)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-org/TransformerDownscaling_MP.git
cd TransformerDownscaling_MP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Install BasicSR Framework

This repo builds on **BasicSR** (registration-based framework for image restoration):

```bash
cd basicsr
pip install -e .
cd ..
```

Key BasicSR components:
- **Registry pattern:** Models, losses, and data loaders registered via decorators
- **Distributed training:** DistributedDataParallel wrapper for multi-GPU setups
- **Logger integration:** Tensorboard, wandb, and local file logging

### 3. Data Preparation

#### Directory Structure

```
data/
├── MSWEP/
│   ├── 20150101.npy       (daily LR precipitation, YYYYMMDD format)
│   ├── 20150102.npy
│   └── ...
├── CHIRPS/
│   ├── 20150101.npy       (daily HR precipitation targets)
│   ├── 20150102.npy
│   └── ...
├── ERA5Land_winds/
│   ├── 20150101.npy       ERA5_winds_2015_*.npy  (u, v components 0.1°)
│   ├── 20150102.npy
│   └── ...
├── SRTM/
│   └── SRTM_India.npy             (static elevation map, ~30 m resampled to 0.1°)
├── masks/
│   ├── land_sea_mask_LR.npy       (0.1° land-sea mask)
│   └── land_sea_mask_HR.npy       (0.05° land-sea mask)
└── more files...
```

#### Download & Preprocess Data

Data preprocessing scripts (regridding, masking, normalization) are in `data_prep/`:

```bash
cd data_prep

# Download MSWEP (example; adjust for your region/period)
python download_mswep.py --start_year 2015 --end_year 2020 --region india --output ../data/MSWEP

# Download CHIRPS (target)
python download_chirps.py --start_year 2015 --end_year 2020 --region india --output ../data/CHIRPS

# Download ERA5-Land winds
python download_era5_winds.py --start_year 2015 --end_year 2020 --output ../data/ERA5Land_winds

# Regrid to common resolution (0.1° LR, 0.05° HR)
# Note: Uses nearest-neighbour interpolation (see interpolation note below)
python regrid_all_data.py --resolution_lr 0.1 --resolution_hr 0.05

# Compute normalization statistics (land pixels only)
python compute_normalization_stats.py --train_years 2015 2016 2017 2018 2019 2020

cd ..
```

**Interpolation Note:** Historical regridding used nearest-neighbour interpolation. Verify against folder naming and YAML config to confirm method.

#### Create Train/Val/Test Splits

```bash
python scripts/create_data_splits.py \
  --data_root ./data \
  --train_years 2015 2016 2017 2018 2019 \
  --val_year 2020_jan_to_jun \
  --test_year 2020_jul_to_dec \
  --output ./data/train_val_test_lists.txt
```

Output: `train_val_test_lists.txt` with format:
```
/path/to/MSWEP_2015_01_01.npy /path/to/CHIRPS_2015_01_01.npy 2015-01-01
/path/to/MSWEP_2015_01_02.npy /path/to/CHIRPS_2015_01_02.npy 2015-01-02
...
```

**Index Chain Warning:** The RadarDataset class chains indices: `idx → data_map[idx] → basename → strip .npy → date token`. Any preprocessing script consuming train/val lists must mirror this exact resolution.

---

## Training Configuration

### Configuration File Structure

Master training YAML: `configs/Uformer_2020_master.yml`

```yaml
# Model architecture
name: 'ClimateUformerMultiscaleFuseModel'
type: ClimateUformerMultiscaleModel
num_in_ch: 2           # MSWEP (LR precip) + ERA5 wind speed (derived from u, v)
num_out_ch: 1          # CHIRPS (HR precip)
embed_dim: 96
depths: [6, 6, 6, 6]
num_heads: [6, 6, 6, 6]
window_size: 16
scale: 2               # 2× upsampling (64→128)
topography_enabled: true
srtm_path: './data/SRTM/SRTM_India.npy'

# Loss function (v2 AsymLoss selected as final)
loss:
  type: 'v2_AsymLoss'  # Toggle: 'v2_AsymLoss' | 'v3_PerPixelPercentile' | 'v4_AsymLoss_SSIM'
  base_loss: 'Charbonnier'
  epsilon: 1.0e-3
  weight_ext: 5.0        # Extreme-event weight ramp
  p90_threshold: 3.1916  # Global P90 threshold
  weight_asymmetry: 2.0  # Under-prediction penalty (parameter a)
  weight_wet: 0.1        # Soft BCE occurrence term

# Training hyperparameters
num_gpu: 2
batch_size: 4          # Per GPU; global batch = 8
num_worker: 4
total_epochs: 120      # Via dataset_enlarge_ratio: 4 (30 real epochs × 4)
dataset_enlarge_ratio: 4
milestone: [20000, 40000]  # Learning rate schedule (steps)
lr_G: 2.0e-4

# Data loader
crop_size: 64          # Input crop (LR)
target_crop_size: 128  # Target crop (HR, 2× input)
normalization: 'log1p_zscore'  # log₁ₚ + z-score (land pixels only)
mask_lr: './data/masks/land_sea_mask_LR.npy'
mask_hr: './data/masks/land_sea_mask_HR.npy'
```

### Running Training on Kaggle

Kaggle sessions have a 30-hour kernel timeout. For 6-year training (estimated ~19 hours), use **MODE B split/resume**:

#### Session 1: Initial Training
```bash
# Copy repo and data to Kaggle working directory
# cd /kaggle/working

# Start training (MODE A: train until kernel timeout)
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  basicsr/train.py \
  -opt configs/Uformer_2020_master.yml
```

#### Session 2: Resume Training
```bash
# MODE B: Resume from checkpoint
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  basicsr/train.py \
  -opt configs/Uformer_2020_master.yml \
  --auto_resume
```

**Checkpoint Management:**
- Checkpoints saved to: `experiments/ClimateUformerMultiscaleFuseModel/models/`
- Latest checkpoint: `latest.pth`
- Best model (by validation loss): `best.pth`
- Resume logic: BasicSR auto-detects `latest.pth` and resumes training state (epoch, LR schedule, etc.)

---

## Model Variants & Loss Functions

Three loss variants were compared on 2020 data (ablation study):

| Variant | Loss Mechanism | Performance | Notes |
|---------|----------------|-------------|-------|
| **v2 AsymLoss** ✓ | Global P90 threshold + asymmetric penalty | Baseline metrics | **Selected for final 6-year training** |
| v3 Per-Pixel Percentile | Per-pixel P90/P10 thresholds | ~same as v2 | Added complexity without improvement |
| v4 AsymLoss + SSIM | SSIM regularization | Structural quality gains, lower intensity | Trade-off: nice structure but lower values |

**Selection Rationale:** v2 provides stable, reproducible extreme-event handling with minimal hyperparameter tuning. Per-pixel percentile thresholds (v3) offered no metric improvement. SSIM regularization (v4) trades mean intensity for structural quality—a consistent pattern across configurations.

---

## Evaluation & Results

### Metrics Computed

On held-out test set (2020-07 to 2020-12, 219 days):

- **Spatial Mean Bias:** Mean absolute error in daily spatial mean precipitation
- **Rainfall Intensity PDF:** Distribution of predicted vs. observed daily rainfall values
- **Extreme Event Indices:** R95p, R99p (95th/99th percentile rainfall amounts)
- **Spatial Structure:** Correlation maps, bias maps (mean predicted − mean observed)

### Key Findings

1. **Systematic Dry Bias Reduction:** v2 AsymLoss achieves ~23% reduction in dry bias vs. Charbonnier baseline
2. **Extended Output Range:** Baseline saturates at ~66 mm/day; v2 AsymLoss reaches ~204 mm/day
3. **Topography Matters:** SRTM branch improves predictions in orographically complex regions
4. **Asymmetric Penalty Critical:** Parameter *a* (under-prediction weight) is the key mechanism; global P90 threshold sufficient for this domain

### Figures

- **Figure 5.2:** LeWin Transformer Block Architecture
- **Figure 6.1:** Asymmetric Loss Weight Distribution
- **Figure 6.2:** P90 Partitioning (Extreme vs. Non-Extreme)
- **Figure 7.1:** Experimental Workflow Flowchart
- **Figure 8.4:** Spatial Comparison (Predicted vs. Observed, Example Day)
- **Figure 8.5:** Mean Precipitation & Bias Maps (All 219 Test Days)
- **Figure 8.6:** Rainfall Intensity PDF with R95p/R99p Inset

---

## Project Structure

```
TransformerDownscaling_MP/
├── basicsr/                    # BasicSR framework (core training logic)
│   ├── archs/
│   │   └── climateformer_arch.py       # ClimateUformerMultiscaleFuseModel
│   ├── data/
│   │   ├── radar_dataset.py            # RadarDataset (train/val/test loader)
│   │   └── data_util.py
│   ├── losses/
│   │   ├── losses.py                   # v2/v3/v4 AsymLoss variants
│   │   └── ssim_loss.py                # Optional SSIM regularization
│   ├── models/
│   │   └── sr_model.py                 # Training loop, backward pass
│   ├── train.py                        # Entry point (distributed training)
│   └── registry.py                     # Decorator-based registration
├── configs/
│   └── Uformer_2020_master.yml         # Master YAML (v2/v3/v4 toggles)
├── data_prep/
│   ├── download_mswep.py
│   ├── download_chirps.py
│   ├── download_era5_winds.py
│   ├── regrid_all_data.py
│   ├── compute_normalization_stats.py
│   └── create_data_splits.py
├── scripts/
│   ├── inference.py                    # Tile-based inference (high-res regions)
│   ├── evaluate.py                     # Compute metrics on test set
│   └── visualize_results.py            # Plot figures
├── experiments/                        # Training outputs
│   └── ClimateUformerMultiscaleFuseModel/
│       ├── models/
│       │   ├── latest.pth
│       │   └── best.pth
│       ├── log/
│       └── tb_logger/                  # Tensorboard logs
├── requirements.txt
└── README.md
```

---

## Usage (High-Level)

### Training

```bash
python -m torch.distributed.launch --nproc_per_node=2 basicsr/train.py -opt configs/Uformer_2020_master.yml
```

### Inference (Tile-Based)

```bash
python scripts/inference.py \
  --model_path experiments/ClimateUformerMultiscaleFuseModel/models/best.pth \
  --config configs/Uformer_2020_master.yml \
  --input_mswep data/MSWEP/2020_07_01.npy \
  --input_era5 data/ERA5Land_winds/2020_07_01.npy \
  --output results/CHIRPS_pred_2020_07_01.npy \
  --tile_size 512
```

### Evaluation

```bash
python scripts/evaluate.py \
  --model_path experiments/ClimateUformerMultiscaleFuseModel/models/best.pth \
  --config configs/Uformer_2020_master.yml \
  --test_list data/test_list.txt \
  --output_dir results/metrics
```

---

## Citation

```bibtex
@mastersthesis{tiwari2024precipitation,
  author = {Tiwari, Vipul},
  title = {Estimating High-Resolution (1-4 km) Daily Precipitation with Space/Time Deep Learning},
  school = {Indian Institute of Technology Bombay},
  year = {2024},
  advisor = {Lanka, Karthikeyan}
}
```

**Architectural Precedent:**

```bibtex
@article{zhong2024investigating,
  author = {Zhong, Y. and others},
  title = {Investigating transformer-based models for spatial downscaling and correcting},
  journal = {Quarterly Journal of the Royal Meteorological Society},
  year = {2024}
}
```

**Comparative Work:**

```bibtex
@article{chandel2025deep,
  author = {Chandel, M. and others},
  title = {Deep Learning Based Statistical Downscaling for Enhanced Representation of Indian},
  journal = {Journal of Geophysical Research: Atmospheres},
  year = {2025}
}
```

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contact

**Vipul Tiwari**  
M.Tech Student, CSRE  
Indian Institute of Technology Bombay  
Roll: 24M0306  
Advisor: Prof. Karthikeyan Lanka

For questions or issues, please open an issue on GitHub or contact [your-email@iitb.ac.in].

---

## Troubleshooting

### Common Issues

**Q: RadarDataset index mismatch during training**  
A: Verify that the date parsing in `data_prep/create_data_splits.py` exactly matches the basename-stripping logic in `basicsr/data/radar_dataset.py`. Both must resolve `MSWEP_2015_01_01.npy` → `2015-01-01`.

**Q: Model saturates at low precipitation values**  
A: Check that the asymmetric penalty parameter *a* is enabled in the loss config. If using baseline Charbonnier, saturation at ~66 mm/day is expected.

**Q: Kaggle kernel timeout during training**  
A: Use MODE B split/resume with checkpoint frequency every 5000 steps. BasicSR's `--auto_resume` flag will detect and restore from the latest checkpoint.

**Q: SRTM topography branch not activating**  
A: Ensure `topography_enabled: true` in YAML and that `srtm_path` points to a valid file. Model will error if the path is missing or inaccessible.
