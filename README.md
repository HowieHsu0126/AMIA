# Neural Granger Causal Discovery for Acute Kidney Injury-Associated Derangements (NGC-AKI)

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-lightgrey)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

---

## 1. Project Overview

This repository hosts the official implementation of our [AMIA 2024 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12099353/) &nbsp;**"Neural Granger Causal Discovery for Acute Kidney Injury-Associated Derangements."**  
The project applies differentiable Granger causality to learn directed temporal graphs among intensive-care variables in the MIMIC-IV database, with a spotlight on derangements related to acute kidney injury (AKI).

---

## 2. Directory Layout

```text
Libs/                     # Core Python library
Models/               # cMLP, cLSTM, cRNN definitions & training loops
Utils/                # Data-processing helpers (preprocessor.py)
Data/SQL/             # Parameterised SQL queries for MIMIC-IV
Input/                    # Raw & processed data (git-ignored)
Output/                   # Saved models, logs, plots (git-ignored)
config/                   # YAML configs for `run_pipeline.py`
NGC.ipynb                 # End-to-end: extraction training evaluation
GC.ipynb                  # Graph visualisation & metrics
Derangement.ipynb         # Clinical interpretation of discovered causes
README.md                 # You are here
```

---

## 3. Quick Start

### 3.1 Create a Python Environment

```bash
# Option A – conda (recommended)
conda create -n ngc_aki python=3.9 -y
conda activate ngc_aki

# Option B – virtualenv
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
# GPU build (CUDA 11.7) – adjust if needed
torch_version="2.0.1+cu117"
pip install torch==${torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Core scientific stack
pip install -r requirements.txt  # OR manually:
# pip install numpy pandas tqdm jupyter seaborn matplotlib scikit-learn
```

### 3.2 Obtain MIMIC-IV

1. Apply for **PhysioNet/MIMIC-IV** access and download the full relational database (PostgreSQL dump).
2. Restore the dump to a local PostgreSQL instance (?12):
   ```bash
   createdb mimiciv
   psql mimiciv < mimic-iv-2.2-postgres.sql
   ```
3. (Optional) Create a `.env` file with connection parameters e.g.
   ```ini
   PGHOST=localhost
   PGUSER=YOUR_USER
   PGPASSWORD=YOUR_PWD
   PGDATABASE=mimiciv
   ```

---

## 4. Data Extraction Pipeline

All SQL files assume the default **`mimiciv`** schema names. To dump raw CSVs:

```bash
cd Libs/Data/SQL
psql -f run.sql          # orchestrates all other scripts
# ??? results saved to ../../../../Input/raw/
```

If you prefer manual execution, run the scripts in the following order:

1. `Concepts/*.sql` – materialise vitals, labs, treatments, scores.
2. `main.sql` – join into a unified, hourly table.
3. `aki.csv` – KDIGO-based AKI cohort, generated within step 2.

---

## 5. Pre-processing

Navigate back to the project root and execute:

```python
from Libs.Utils.preprocessor import (
    resample_and_mask,
    find_stay_id_intersection,
    merge_csv_based_on_aki_id,
    reduce_mem_usage,
)

# 1 – harmonise patient lists across tables
find_stay_id_intersection(
    directory_path="Input/raw/",
    output_file_path="Input/processed/AKI_ID.csv",
)

# 2 – merge individual tables into a single multivariate series
merge_csv_based_on_aki_id(
    directory_path="Input/raw/",
    aki_id_file_path="Input/processed/AKI_ID.csv",
    output_file_path="Input/processed/dataset.csv",
)

# 3 – hourly resampling & masking (24 h window shown)
resample_and_mask(
    input_file_path="Input/processed/dataset.csv",
    output_file_path="Input/processed/dataset_1h.csv",
    verbose=True,
)

# 4 – memory footprint optimisation (optional)
reduce_mem_usage(
    input_path="Input/processed/dataset_1h.csv",
    output_path="Input/processed/dataset_1h_optimized.csv",
)
```

_Tip:_ adjust the resampling horizon (`iloc[-24:]`) inside `preprocessor.py` to use longer contexts.

Convert the final CSV to a PyTorch tensor

```python
import pandas as pd, torch

df = pd.read_csv("Input/processed/dataset_1h_optimized.csv")
X  = torch.tensor(df.values, dtype=torch.float32)
# reshape  (batch, time, p)
X  = X.view(-1, 24, X.shape[-1])

torch.save(X, "Input/processed/tensor.pt")
```

Alternatively you can store every hyper-parameter in a single YAML file and let
the helper automatically orchestrate *all* steps (SQL ? preprocessing ?
training):

```bash
# default config lives at config/train.yml – edit as required
python -m Libs.run_pipeline --config config/train.yml  # add --cpu to disable CUDA
```

If you want to access the config programmatically:

```python
from Libs.Utils.config import load_config

cfg = load_config("config/train.yml")
print(cfg["model"]["backbone"], cfg["model"]["lag"], cfg["model"]["training"]["lr"])
```

---

## 6. Model Training

```python
import torch
from Libs.Models.cMLP import cMLP, train_model_gista

X = torch.load("Input/processed/tensor.pt")
model = cMLP(num_series=X.shape[-1], lag=24, hidden=[128, 64])
train_model_gista(
    cmlp       = model,
    X          = X,
    lam        = 1e-2,   # sparsity weight
    lam_ridge  = 1e-3,   # ridge penalty
    lr         = 1e-3,
    penalty    = "H",    # {H, GL, GSGL}
    max_iter   = 5000,
    verbose    = 1,
)

graph = model.GC(threshold=True)  # Boolean p×p adjacency matrix
torch.save(graph, "Output/GC_matrix.pt")
```

For LSTM or RNN back-bones replace the import with `cLSTM` or `cRNN`.

---

## 7. Evaluation & Visualisation

Run notebook **`GC.ipynb`** to compute graph-level metrics (precision/recall, SHD, etc.) and render chord diagrams.  
Run **`Derangement.ipynb`** for clinical interpretation and outcome association analyses.

---

## 8. Reproducing Paper Figures

```bash
jupyter notebook NGC.ipynb   # executes all steps sequentially
```

Expected outputs will populate the `Output/` directory:

- `GC_matrix.pt` – learned adjacency matrix.
- `training_log.csv` – loss & sparsity curves.
- `fig_*` PNG/PDF files – figures used in the manuscript.

---

## 9. Troubleshooting

| Issue                         | Possible Cause                            | Fix                                                      |
| ----------------------------- | ----------------------------------------- | -------------------------------------------------------- |
| **CUDA out-of-memory**        | Default batch size too large              | Reduce `X` batch dimension or use CPU version.           |
| **psql: permission denied**   | Database role lacks privileges            | Grant SELECT on MIMIC-IV schema or connect as superuser. |
| **UnicodeDecodeError** on CSV | MIMIC export uses `windows-1252` encoding | Add `encoding="latin-1"` flag in `pd.read_csv`.          |

---

## 10. Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{xu2024ngcaki,
  title     = {Neural Granger Causal Discovery for Acute Kidney Injury-Associated Derangements},
  author    = {Haowei Xu, Wentie Liu, Tongyue Shi and Guilan Kong},
  booktitle = {AMIA Annual Symposium},
  year      = {2024}
}
```

---

## 11. License

This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 12. Contact

For questions, please open an issue or email **haoweixu@stu.pku.edu.cn**.
