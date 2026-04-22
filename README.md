# Non-contact Plant/Soil Moisture Monitoring (Baseline MLP Regression)

This repository provides a **minimal, complete, and runnable** baseline machine learning project for:

> **Non-contact plant/soil moisture monitoring based on microwave transmission and edge AI**

The current version uses a **synthetic dataset** to validate the full pipeline before replacing it with real measured data.

---

## Important data disclaimer

The generated dataset is **synthetic data for pipeline validation/debugging only**.  
It does **not** represent real calibrated experimental measurements.  
For real deployment or scientific conclusions, replace this dataset with real collected and calibrated data.

---

## Project structure

```text
.
├── data/
│   └── (generated CSV will be saved here)
├── outputs/
│   └── (best checkpoint and plots will be saved here)
├── dependency_bootstrap.py     # auto-install missing libs helper
├── run_pipeline.py             # one-click pipeline runner
├── generate_data.py
├── model.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Synthetic data design (physical intuition)

The synthetic generator follows these assumptions:

1. **Main relation:** moisture affects microwave attenuation, so detector **voltage generally decreases as moisture increases**.
2. **Environmental disturbance:** `temperature` and `humidity` affect both moisture distribution and voltage drift.
3. **Nonlinearity:** sinusoidal and interaction terms are added.
4. **Noise:** Gaussian noise is injected into voltage and moisture to mimic measurement uncertainty.

Generated columns:
- `voltage`
- `temperature`
- `humidity`
- `moisture`

---

## Automatic dependency installation (new)

If your computer is missing some Python libraries, the project now supports **auto-install before running**:

- `generate_data.py` will auto-check/install `numpy`, `pandas`
- `train.py` will auto-check/install `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`
- `run_pipeline.py` provides one-click full process (install deps → generate data → train/evaluate)

This makes the baseline friendlier for beginner machines.

---

## Model and training setup

- Framework: **PyTorch**
- Task: **Regression**
- Model: **MLP** with linear output layer
- Loss: **MSELoss**
- Optimizer: **Adam**
- Split: 70% train / 15% val / 15% test
- Scaling: `StandardScaler` fitted **only on training set**
- Early stopping: enabled via patience + best checkpoint saving
- Test metrics:
  - MAE
  - RMSE
  - R²

Future feature extension placeholders are easy to add, e.g.:
- `soil_type`
- `distance`
- `baseline_voltage`

---

## GPU / CPU behavior

Training automatically uses GPU when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The code:
1. Moves model to `device`
2. Moves every batch (train/val/test) to `device`
3. Computes loss/inference on the same device
4. Saves checkpoint that can be loaded on either GPU or CPU using `map_location`

So GPU is preferred, but CPU fallback is fully supported.

---

## Environment setup (recommended but optional)

```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# Windows CMD:
# .venv\Scripts\activate.bat
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Even if you skip manual install, scripts can auto-install missing packages.

### Quick CUDA visibility check

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## How to run

### Option A (recommended): one-click full pipeline

```bash
python run_pipeline.py --n-samples 6000 --seed 42 --data-path data/synthetic_moisture.csv --output-dir outputs --epochs 120 --batch-size 128 --learning-rate 1e-3 --patience 18
```

### Option B: run step-by-step

1) Generate synthetic dataset

```bash
python generate_data.py --n-samples 6000 --seed 42 --output data/synthetic_moisture.csv
```

2) Train/evaluate baseline MLP

```bash
python train.py --data-path data/synthetic_moisture.csv --output-dir outputs --epochs 120 --batch-size 128 --learning-rate 1e-3 --patience 18
```

---

## Expected outputs

After training, you should see:

- `outputs/best_model.pt` (best checkpoint)
- `outputs/loss_curve.png` (train/val loss curves)
- `outputs/pred_vs_true.png` (predicted vs true scatter)

Terminal logs include:
- device information (CUDA availability, selected device, GPU name if used)
- missing value checks
- per-epoch training/validation losses
- final MAE, RMSE, R²
- short convergence + overfit/underfit summary

---

## How to open this project in VS Code (Windows)

### Method 1: from VS Code UI
1. Open VS Code.
2. Click **File** → **Open Folder...**
3. Select your project folder (this repository root).
4. Open terminal in VS Code (`Ctrl + \``).
5. (Optional) Activate venv, then run:
   ```powershell
   python run_pipeline.py
   ```

### Method 2: from PowerShell
1. Open PowerShell and `cd` to project folder.
2. Run:
   ```powershell
   code .
   ```
3. In VS Code terminal, run:
   ```powershell
   python run_pipeline.py
   ```

> If `code .` is not recognized, in VS Code press `Ctrl+Shift+P` and run:  
> **Shell Command: Install 'code' command in PATH** (or reinstall VS Code with PATH option).

---

## Beginner-friendly Windows + NVIDIA GPU step-by-step

1. Open **PowerShell** in the project folder.
2. Create and activate venv:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. (Optional) Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Verify PyTorch can see CUDA:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
   ```
5. Run full project with one command:
   ```powershell
   python run_pipeline.py
   ```
6. Open the `outputs/` folder and check:
   - `best_model.pt`
   - `loss_curve.png`
   - `pred_vs_true.png`

If CUDA is not available, the script will run on CPU automatically.

---

## Notes for next stage

Once the pipeline is verified, replace synthetic CSV with real STM32 + detector data and revisit:
- feature engineering
- calibration strategy
- model selection
- deployment constraints for edge AI
