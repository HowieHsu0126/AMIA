"""End-to-end pipeline runner for NGC-AKI.

Usage (command line):
    python -m Libs.run_pipeline --config config/train.yml --cpu

The YAML configuration controls the following high-level stages:

```yaml
sql:
  enabled: true                      # execute SQL extraction
  psql_bin: psql                     # path to `psql` client
  conn:                               # passed as env vars or psql flags
    host: localhost
    port: 5432
    user: postgres
    db: mimiciv
  script: Libs/Data/SQL/run.sql      # top-level orchestration script

preprocess:
  enabled: true
  raw_dir: Input/raw/
  processed_dir: Input/processed/
  resample_horizon: 24               # hours

model:
  backbone: cMLP                     # {cMLP,cRNN,cLSTM}
  lag: 24
  hidden: [128, 64]
  training:
    optimizer: gista                 # {gista,adam,ista}
    max_iter: 5000
    lr: 1e-3
    lam: 1e-2
    lam_ridge: 1e-3
    penalty: H
    batch_size: 32                   # reserved for future use
```

The script intentionally keeps external dependencies minimal: only *psycopg2* is
needed for optional in-Python SQL execution; otherwise we fallback to the `psql`
CLI tool.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import torch

from Libs.Utils.config import load_config
from Libs.Utils import preprocessor as prep
from Libs.Models import cMLP, cRNN, cLSTM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silence noisy warnings (FutureWarning, UserWarning, etc.) -------------------
# ---------------------------------------------------------------------------
# These warnings tend to clutter the console/logs and are not actionable for
# end-users in the context of running the pipeline.  They can still be
# re-enabled by commenting out the following line if needed for debugging.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_shell(cmd: str, env: Dict[str, str] | None = None) -> None:
    """Run *cmd* in a subprocess, forwarding output to the parent terminal."""
    logger.info("$ %s", cmd)
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        logger.error("Command failed with return code %s", exc.returncode)
        raise


# ---------------------------------------------------------------------------
# Pipeline stages ------------------------------------------------------------
# ---------------------------------------------------------------------------

def stage_sql(cfg: Dict[str, Any]) -> None:
    if not cfg.get("enabled", False):
        logger.info("[SQL] Stage disabled – skipping")
        return

    script_path = Path(cfg["script"]).expanduser()
    if not script_path.is_file():
        raise FileNotFoundError(script_path)

    psql_bin = cfg.get("psql_bin", "psql")
    conn = cfg.get("conn", {})

    # Build connection string flags, e.g. -h host -p port -U user -d db
    conn_flags = []
    mapping = {
        "host": "-h",
        "port": "-p",
        "user": "-U",
        "db": "-d",
    }
    for key, flag in mapping.items():
        val = conn.get(key)
        if val is not None:
            conn_flags.extend([flag, str(val)])

    # ------------------------------------------------------------------
    # Inject DATA_DIR variable for export path substitution -------------
    # ------------------------------------------------------------------
    data_dir = cfg.get("data_dir", "Input/raw/")  # Fallback keeps legacy default
    cmd = (
        f"{psql_bin} {' '.join(conn_flags)} -v ON_ERROR_STOP=1 "
        f"-v DATA_DIR='{data_dir}' -f {script_path}"
    )
    run_shell(cmd, env=os.environ)


def stage_preprocess(cfg: Dict[str, Any]) -> None:
    if not cfg.get("enabled", False):
        logger.info("[Preprocess] Stage disabled – skipping")
        return

    raw_dir = Path(cfg["raw_dir"]).expanduser()
    processed_dir = Path(cfg["processed_dir"]).expanduser()
    processed_dir.mkdir(parents=True, exist_ok=True)

    # If artefacts already exist, allow skipping the entire preprocessing step
    final_csv_marker = processed_dir / "dataset_reduced.csv"
    tensor_marker = processed_dir / "tensor.pt"
    if final_csv_marker.exists() and tensor_marker.exists():
        logger.info(
            "[Preprocess] Detected existing preprocessed files (%s, %s) – skipping preprocessing stage.",
            final_csv_marker.name,
            tensor_marker.name,
        )
        return

    # ------------------------------------------------------------------
    # File naming convention expected by downstream consumers ------------
    # ------------------------------------------------------------------
    merged_csv = processed_dir / "data_merged.csv"
    dataset_csv = processed_dir / "dataset.csv"  # after AKI filtering
    dataset_filtered_csv = processed_dir / "dataset_filtered.csv"
    dataset_filled_csv = processed_dir / "dataset_filled.csv"
    dataset_derangement_csv = processed_dir / "dataset_derangement.csv"
    dataset_reduced_csv = processed_dir / "dataset_reduced.csv"

    # ------------------------------------------------------------------
    # 1) Merge all raw tables -------------------------------------------
    # ------------------------------------------------------------------
    prep.merge_all_csv(raw_dir, merged_csv)

    # ------------------------------------------------------------------
    # 2) Filter by AKI patients ----------------------------------------
    # ------------------------------------------------------------------
    aki_csv = raw_dir / "aki.csv"
    if not aki_csv.is_file():
        raise FileNotFoundError(aki_csv)
    prep.merge_csv_based_on_aki_id(
        directory_path=raw_dir,
        aki_id_file_path=aki_csv,
        output_file_path=dataset_csv,
    )

    # 2b) Filter out patients with < min_records (configurable, default 4) --
    from Libs.Utils import preprocessor as _prep  # avoid name shadowing
    min_records = cfg.get("min_records", 4)
    _prep.filter_by_min_records(dataset_csv, dataset_filtered_csv, min_records=min_records)

    # ------------------------------------------------------------------
    # 3) Fill missing values -------------------------------------------
    # ------------------------------------------------------------------
    prep.fill_missing_values(dataset_filtered_csv, dataset_filled_csv)

    # ------------------------------------------------------------------
    # 4) Derangement/resampling step -----------------------------------
    # ------------------------------------------------------------------
    prep.resample_and_mask(
        input_file_path=dataset_filled_csv,
        output_file_path=dataset_derangement_csv,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 5) Derangement flag engineering ----------------------------------
    # ------------------------------------------------------------------
    try:
        prep.add_derangement_flags(dataset_derangement_csv, dataset_derangement_csv)
    except Exception as exc:
        logger.warning("Derangement flag generation failed: %s", exc)

    # ------------------------------------------------------------------
    # 6) Memory optimisation -------------------------------------------
    # ------------------------------------------------------------------
    prep.reduce_mem_usage(dataset_derangement_csv, dataset_reduced_csv)

    # ------------------------------------------------------------------
    # 7) Convert final CSV ➜ Tensor ------------------------------------
    # ------------------------------------------------------------------
    import pandas as pd
    import numpy as np

    df = pd.read_csv(dataset_reduced_csv)

    # Drop identifier columns that should not be treated as input features
    for id_col in ["patient_id", "stay_id", "admission_id", "hadm_id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    # Ensure numerical tensor – coerce non‐numeric to NaN then fill with 0
    df_numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (df_numeric.dtypes == np.object_).any():
        # As a last safeguard, drop non-numeric columns
        df_numeric = df_numeric.select_dtypes(include=[np.number])

    X = torch.tensor(df_numeric.values, dtype=torch.float32)

    horizon = cfg.get("resample_horizon", 24)
    if X.shape[0] % horizon != 0:
        logger.info(
            "Number of rows (%d) not divisible by horizon (%d) – truncating",
            X.shape[0], horizon,
        )
        X = X[: X.shape[0] - (X.shape[0] % horizon)]

    X = X.view(-1, horizon, X.shape[-1])
    tensor_path = processed_dir / "tensor.pt"
    torch.save(X, tensor_path)
    logger.info("Saved preprocessed tensor ➜ %s", tensor_path)


def build_model(cfg: Dict[str, Any], num_series: int):
    backbone = cfg["backbone"].lower()
    lag = cfg["lag"]
    hidden = cfg.get("hidden", [128, 64])

    if backbone == "cmlp":
        return cMLP.cMLP(num_series, lag, hidden)
    if backbone == "crnn":
        return cRNN.cRNN(num_series, hidden)
    if backbone == "clstm":
        return cLSTM.cLSTM(num_series, hidden)

    raise ValueError(f"Unsupported backbone: {backbone}")


def stage_training(cfg: Dict[str, Any], tensor_path: Path, device: torch.device):
    X = torch.load(tensor_path)
    X = X.to(device)

    # Ensure the specified lag/context is compatible with tensor length
    time_steps = X.shape[1]
    if cfg.get("lag", 1) >= time_steps:
        new_lag = max(1, time_steps - 1)
        logger.info(
            "Configured lag (%d) >= available time steps (%d) – reducing lag to %d",
            cfg.get("lag"), time_steps, new_lag,
        )
        cfg["lag"] = new_lag

    model = build_model(cfg, num_series=X.shape[-1])
    model = model.to(device)

    train_cfg = cfg["training"]
    optim = train_cfg.get("optimizer", "gista").lower()

    if isinstance(model, cMLP.cMLP):
        if optim == "gista":
            cMLP.train_model_gista(
                model, X,
                lam=train_cfg["lam"],
                lam_ridge=train_cfg["lam_ridge"],
                lr=train_cfg["lr"],
                penalty=train_cfg["penalty"],
                max_iter=train_cfg["max_iter"],
            )
        elif optim == "adam":
            cMLP.train_model_adam(
                model, X,
                lr=train_cfg["lr"],
                max_iter=train_cfg["max_iter"],
                lam=train_cfg["lam"],
                lam_ridge=train_cfg["lam_ridge"],
                penalty=train_cfg["penalty"],
                batch_size=train_cfg.get("batch_size"),
            )
        else:
            raise ValueError(optim)

    elif isinstance(model, (cRNN.cRNN, cLSTM.cLSTM)):
        context = cfg["lag"]  # reuse lag as context length
        if optim == "gista" and isinstance(model, cRNN.cRNN):
            cRNN.train_model_gista(
                model, X,
                context=context,
                lam=train_cfg["lam"],
                lam_ridge=train_cfg["lam_ridge"],
                lr=train_cfg["lr"],
                max_iter=train_cfg["max_iter"],
            )
        else:
            # Fall back to Adam
            train_fn = cRNN.train_model_adam if isinstance(model, cRNN.cRNN) else cLSTM.train_model_adam
            train_fn(
                model, X,
                context=context,
                lr=train_cfg["lr"],
                max_iter=train_cfg["max_iter"],
                lam=train_cfg["lam"],
                lam_ridge=train_cfg["lam_ridge"],
                batch_size=train_cfg.get("batch_size"),
            )

    # Persist outputs --------------------------------------------------------
    output_dir = Path("Output")
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    gc_tensor = model.GC(threshold=False).cpu()  # save raw norms, not binary
    gc_path = output_dir / "GC_matrix.pt"
    torch.save(gc_tensor, gc_path)

    # ------------------------------------------------------------------
    # Optional: auto-export GraphML for downstream visualisation --------
    # ------------------------------------------------------------------
    try:
        from Libs.Utils.gc_graph import export_graphml

        graphml_path = export_graphml(gc_path, output_dir=output_dir / "Results")
        logger.info("GC GraphML exported ➜ %s", graphml_path)
    except Exception as exc:
        logger.warning("GraphML export failed: %s", exc)

    logger.info("Training complete – artifacts saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Entry-point ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full NGC-AKI pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Reproducibility & device handling ---------------------------------
    # ------------------------------------------------------------------
    seed = cfg.get("seed")
    if seed is not None:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Global seed set to %d", seed)

    device_str = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")).lower()
    device = torch.device(device_str if device_str != "cuda" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA requested in config but not available – falling back to CPU")
        device = torch.device("cpu")

    # CLI flag --cpu overrides YAML device
    if args.cpu:
        logger.info("Running on CPU as per configuration/flag")
    else:
        logger.info("Running on device: %s", device)

    # ------------------------------------------------------------------
    # Resolve global I/O paths defined under cfg["paths"] --------------
    # ------------------------------------------------------------------
    paths_cfg = cfg.setdefault("paths", {})
    raw_dir = paths_cfg.setdefault("raw_dir", "Input/raw/")
    processed_dir = paths_cfg.setdefault("processed_dir", "Input/processed/")
    sql_output_dir = paths_cfg.setdefault("sql_output_dir", raw_dir)

    # Back-propagate resolved paths to stage-specific configs for legacy
    # compatibility in case user config still uses older schema.
    cfg.setdefault("preprocess", {}).setdefault("raw_dir", raw_dir)
    cfg["preprocess"].setdefault("processed_dir", processed_dir)
    cfg.setdefault("sql", {}).setdefault("data_dir", sql_output_dir)

    stage_sql(cfg.get("sql", {}))
    stage_preprocess(cfg.get("preprocess", {}))

    tensor_path = Path(cfg["preprocess"]["processed_dir"]).expanduser() / "tensor.pt"
    if not tensor_path.exists():
        logger.error("Tensor file not found: %s – did preprocessing succeed?", tensor_path)
        sys.exit(1)

    stage_training(cfg["model"], tensor_path, device=device)


if __name__ == "__main__":
    main() 