"""Data preprocessing utilities for the NGC-AKI project.

This module provides helpers for resampling, masking, memory‐optimisation and
CSV merging. All public functions follow Google-style docstrings and rely on
the :pymod:`logging` package instead of ``print`` for status messages so that
users can globally configure verbosity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def resample_and_mask(
    input_file_path: Optional[Union[str, Path]] = None,
    output_file_path: Optional[Union[str, Path]] = None,
    *,
    verbose: bool = False,
    **kwargs,
) -> None:
    """Resample an hourly time-series table and generate value/mask pairs.

    The function expects a **long table** with the following columns::

        stay_id | hour | <feature_1> | <feature_2> | ...

    It converts the table to a **wide, hourly-binned** representation, performs
    forward/backward propagation of missing values, and finally writes the
    processed CSV to *output_file_path*.

    Args:
        input_file_path: Path to the raw CSV file.
        output_file_path: Destination path for the processed CSV.
        verbose: If ``True`` log progress messages at ``INFO`` level, else
            ``DEBUG``.

    Returns:
        ``None`` – the processed dataset is written to *output_file_path*.
    """

    # ------------------------------------------------------------------
    # Backward-compatibility layer --------------------------------------
    # ------------------------------------------------------------------
    # Allow legacy calls that erroneously used the *iutput_file_path* keyword.
    if input_file_path is None and "iutput_file_path" in kwargs:
        logger.warning("`iutput_file_path` is deprecated; use `input_file_path` instead.")
        input_file_path = kwargs.pop("iutput_file_path")

    if input_file_path is None:
        raise ValueError("`input_file_path` must be provided.")

    if output_file_path is None:
        raise ValueError("`output_file_path` must be provided.")

    input_file_path = Path(input_file_path)

    level = logging.INFO if verbose else logging.DEBUG
    _log = logger.log

    _log(level, "Reading input CSV: %s", input_file_path)
    timeseries = pd.read_csv(input_file_path)
    if timeseries.empty:
        logger.warning("Input CSV is empty: %s", input_file_path)

    if verbose:
        _log(level, "Resampling to 1-hour intervals …")
    # Identify the ID column dynamically (stay_id, admission_id, hadm_id)
    id_col = None
    for cand in ["stay_id", "admission_id", "hadm_id"]:
        if cand in timeseries.columns:
            id_col = cand
            break
    if id_col is None:
        raise KeyError("No ID column (stay_id/admission_id/hadm_id) found in input CSV")

    # ------------------------------------------------------------------
    # Per-patient resampling to avoid huge wide pivot --------------------
    # ------------------------------------------------------------------
    # Ensure 'hour' column is integer for safe indexing
    timeseries['hour'] = (
        pd.to_numeric(timeseries['hour'], errors='coerce')
        .fillna(0)
        .astype(int)
    )

    timeseries = (
        timeseries
        .groupby([id_col, 'hour'], as_index=False)
        .mean()
    )

    feature_cols = [c for c in timeseries.columns if c not in (id_col, 'hour')]
    horizon = kwargs.get('horizon', 24)

    outputs = []
    for pid, sub in timeseries.groupby(id_col, sort=False):
        sub = sub.set_index('hour').sort_index()

        # Build continuous hourly index covering observed span
        idx = pd.RangeIndex(int(sub.index.min()), int(sub.index.max()) + 1)
        sub = sub.reindex(idx)

        # Forward/backward fill then truncate to final `horizon` rows
        sub_ff = sub.fillna(method='ffill').fillna(method='bfill').tail(horizon)

        sub_ff[id_col] = pid
        sub_ff['hour'] = sub_ff.index
        outputs.append(sub_ff.reset_index(drop=True))

    if not outputs:
        logger.warning("No data after resampling – output skipped: %s", output_file_path)
        return

    resampled = pd.concat(outputs, ignore_index=True)[[id_col, 'hour'] + feature_cols]

    resampled.fillna(0, inplace=True)

    _log(level, "Writing resampled CSV to: %s", output_file_path)
    resampled.to_csv(output_file_path, index=False)


def find_stay_id_intersection(directory_path: Union[str, Path], output_file_path: Union[str, Path]) -> None:
    """
    在指定目录下找到所有CSV文件中stay_id列的交集，并保存为CSV文件。

    参数:
    directory_path (str): 包含CSV文件的目录路径。
    output_file_path (str): 输出CSV文件的完整路径。
    """
    directory_path = Path(directory_path)
    output_file_path = Path(output_file_path)

    csv_files = [f for f in directory_path.glob('*.csv')]

    if not csv_files:
        logger.warning("目录中没有CSV文件: %s", directory_path)
        return

    # Helper to identify ID column -------------------------------------------------
    def _get_id_column(df: pd.DataFrame) -> str:
        candidates = ["stay_id", "admission_id", "hadm_id"]
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(
            f"None of the expected ID columns {candidates} found in {df.columns.tolist()}"
        )

    # 读取第一个CSV文件并初始化交集集合
    first_df = pd.read_csv(csv_files[0])
    id_col = _get_id_column(first_df)
    intersection_set = set(first_df[id_col])

    # 遍历其余的CSV文件
    for file_path in csv_files[1:]:
        df = pd.read_csv(file_path)
        id_col_curr = _get_id_column(df)
        intersection_set = intersection_set.intersection(set(df[id_col_curr]))

    # 将交集集合转换为DataFrame
    aki_id_df = pd.DataFrame(list(intersection_set), columns=[id_col])

    # 保存为CSV文件
    aki_id_df.to_csv(output_file_path, index=False)

    # 打印交集集合
    logger.info("Final number of patients after intersection: %d", len(intersection_set))

# # 使用示例
# directory_path = '/home/hwxu/Projects/Dataset/PKU/AMIA/Input/raw/'
# output_file_path = '/home/hwxu/Projects/Dataset/PKU/AMIA/Input/processed/AKI_ID.csv'
# find_stay_id_intersection(directory_path, output_file_path)

def merge_csv_based_on_aki_id(
    directory_path: Union[str, Path],
    aki_id_file_path: Union[str, Path],
    output_file_path: Union[str, Path],
) -> None:
    """
    根据AKI_ID.csv中的stay_id，重新合并目录下的所有CSV文件。

    参数:
    directory_path (str): 包含CSV文件的目录路径。
    aki_id_file_path (str): AKI_ID.csv文件的路径，包含过滤后的stay_id。
    output_file_path (str): 合并后的CSV文件的保存路径。
    """
    aki_id_df = pd.read_csv(aki_id_file_path)

    def _get_id_column(df: pd.DataFrame) -> str:
        candidates = ["stay_id", "admission_id", "hadm_id"]
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError("ID column not found in AKI_ID file")

    # Build mapping of ID column ➜ allowed values --------------------------
    id_sets: dict[str, set] = {}
    for cand in ["stay_id", "hadm_id", "admission_id"]:
        if cand in aki_id_df.columns:
            id_sets[cand] = set(aki_id_df[cand].dropna())

    if not id_sets:
        raise KeyError("No recognised ID columns found in AKI label file")

    # 初始化一个空的DataFrame用于合并数据
    merged_df = pd.DataFrame()

    # 列出目录下的所有CSV文件
    directory_path = Path(directory_path)
    csv_files = [f for f in directory_path.glob('*.csv')]
    logger.info("Files to be processed: %s", csv_files)

    # 遍历CSV文件
    for file_path in csv_files:
        # Skip the label file itself if present in *directory_path*
        if file_path.name.lower() == "aki.csv":
            logger.debug("Skipping label file during data merge: %s", file_path)
            continue
        df = pd.read_csv(file_path)
        
        # 过滤数据，只保留存在于 AKI 标签文件中相对应 ID 列的行
        id_col_curr = _get_id_column(df)
        if id_col_curr in id_sets:
            target_ids = id_sets[id_col_curr]
        # Treat admission_id and hadm_id as synonyms – fall back accordingly
        elif id_col_curr == "admission_id" and "hadm_id" in id_sets:
            target_ids = id_sets["hadm_id"]
        elif id_col_curr == "hadm_id" and "admission_id" in id_sets:
            target_ids = id_sets["admission_id"]
        else:
            logger.debug(
                "ID column %s not present in AKI file – skipping file %s", id_col_curr, file_path
            )
            continue
        filtered_df = df[df[id_col_curr].isin(target_ids)]

        # 将过滤后的数据添加到合并的DataFrame中
        merged_df = pd.concat([merged_df, filtered_df], ignore_index=True)
    
    if 'subject_id' in merged_df.columns:
        merged_df.drop(['subject_id'], axis=1, inplace=True)
    merged_df.rename(columns={'Unnamed: 1': 'time'}, inplace=True)

    # 如果结果为空则直接报错，避免生成 0 字节文件
    if merged_df.empty:
        raise ValueError(
            "No matching records found between raw CSVs and AKI label file. "
            "Please verify that identifier columns align (stay_id / hadm_id / admission_id)."
        )

    # 保存合并后的DataFrame到CSV
    merged_df.to_csv(output_file_path, index=False)
    
# # 示例用法
# directory_path = '/home/hwxu/Projects/Dataset/PKU/AMIA/Input/processed/'
# aki_id_file_path = '/home/hwxu/Projects/Dataset/PKU/AMIA/Input/processed/AKI_ID.csv'
# output_file_path = '/home/hwxu/Projects/Dataset/PKU/AMIA/Input/filtered/dataset.csv'
# merge_csv_based_on_aki_id(directory_path, aki_id_file_path, output_file_path)

def fill_missing_values(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    # 加载数据
    data = pd.read_csv(input_path)
    columns_to_fill = data.columns

    # 应用线性插值
    data_filled = data[columns_to_fill].interpolate(method='linear')

    # 使用向前填充处理插值后仍然存在的缺失值
    data_filled = data_filled.ffill()

    # 对于序列开始就缺失的数据，使用向后填充
    data_filled = data_filled.bfill()

    # 去除重复的列，假设重复是指列名完全相同
    data_filled = data_filled.loc[:,~data_filled.columns.duplicated()]

    # 保存填充后的数据集
    data_filled.to_csv(output_path, index=False)
    
    logger.info("Processed data saved to %s", output_path)
    
# # 使用示例
# input_path = '/path/to/your/dataset.csv'  # 请替换为实际的数据集路径
# output_path = '/path/to/your/dataset_filled.csv'  # 请替换为你想保存填充后数据集的路径

# # 调用函数填充缺失值并保存结果
# fill_missing_values(input_path, output_path)

def reduce_mem_usage(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    verbose: bool = True,
) -> None:
    df = pd.read_csv(input_path)
    numerics = ['int16', 'int32', 'int64', 
                'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            "Mem. usage decreased to %.2f Mb (%.1f%% reduction)",
            end_mem,
            100 * (start_mem - end_mem) / start_mem,
        )
    # 保存填充后的数据集
    df.to_csv(output_path, index=False)
    
    logger.info("Processed data saved to %s", output_path)

def merge_all_csv(
    directory_path: Union[str, Path],
    output_file_path: Union[str, Path],
    *,
    verbose: bool = True,
) -> None:
    """Merge **all** CSV files within *directory_path* into a single file.

    The function performs an outer concatenation (``pd.concat``) of all CSVs
    located directly under *directory_path* (non-recursive) and writes the
    resulting table to *output_file_path*. Duplicate rows (based on *all*
    columns) are removed while preserving the original order.

    Args:
        directory_path: Folder that contains the source ``.csv`` files.
        output_file_path: Destination path for the merged CSV.
        verbose: If ``True`` log an ``INFO`` message for each processed file,
            else defer to ``DEBUG``.

    Returns:
        ``None`` – the merged dataset is persisted to *output_file_path*.
    """

    directory_path = Path(directory_path)
    output_file_path = Path(output_file_path)

    if not directory_path.is_dir():
        raise NotADirectoryError(directory_path)

    csv_files = sorted(directory_path.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in directory: %s", directory_path)
        return

    log_level = logging.INFO if verbose else logging.DEBUG
    _log = logger.log

    # Keys we expect in all per-concept files
    key_candidates = ["stay_id", "hadm_id", "admission_id", "hour"]

    base_df: pd.DataFrame | None = None
    for fp in csv_files:
        if fp.name.lower() == "aki.csv":
            _log(log_level, "Skipping label file: %s", fp)
            continue

        _log(log_level, "Reading %s", fp)
        df = pd.read_csv(fp)

        # Determine join keys present in this DF
        join_keys = [c for c in key_candidates if c in df.columns]
        if "hour" not in df.columns:
            raise KeyError(f"File {fp} is missing mandatory key 'hour'.")

        feature_cols = [c for c in df.columns if c not in join_keys]

        # Ensure unique feature names across files by suffixing with filename stem when needed
        col_rename = {}
        for fc in feature_cols:
            if base_df is not None and fc in base_df.columns:
                col_rename[fc] = f"{fc}_{fp.stem}"
        if col_rename:
            df = df.rename(columns=col_rename)

        if base_df is None:
            base_df = df
        else:
            # Determine common join keys between current base and new frame
            common_keys = ["hour"] + [k for k in key_candidates if k in base_df.columns and k in df.columns]
            base_df = base_df.merge(df, on=common_keys, how="outer")

    if base_df is None:
        logger.warning("No data merged – aborting.")
        return

    # Drop duplicate rows (same keys & identical features)
    before = len(base_df)
    base_df = base_df.drop_duplicates()
    after = len(base_df)
    _log(log_level, "Final merged shape: %s (deduped %.1f%% rows)", base_df.shape, 100*(before-after)/before if before else 0)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_csv(output_file_path, index=False)

    logger.info("Merged CSV saved to %s", output_file_path)