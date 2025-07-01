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
    # Setting the 'hour' column as index along with 'stay_id'
    timeseries.set_index(['stay_id', 'hour'], inplace=True)

    # Take the mean of any duplicate index entries for unstacking
    timeseries = timeseries.groupby(level=[0, 1]).mean()
    # Put patient into columns so that we can round the timedeltas to the nearest hour and take the mean in the time interval
    unstacked = timeseries.unstack(level=0)
    unstacked.index = pd.to_timedelta(unstacked.index, unit='H').ceil('H')
    resampled = unstacked.resample('H').mean()

    if verbose:
        _log(level, "Filling missing data forwards and then backwards...")
    # First carry forward missing values, then carry backward
    resampled = resampled.fillna(method='ffill').fillna(
        method='bfill').iloc[-24:]

    # Simplify the indexes of both tables
    resampled.index = list(range(1, 25))

    if verbose:
        _log(level, "Filling in remaining values with zeros...")
    # This step might not be necessary if the above fillna methods cover all cases,
    # but it's a good safeguard.
    resampled.fillna(0, inplace=True)

    if verbose:
        _log(level, "Reconfiguring and combining features with mask features...")
    # Pivot the table around to give the final data
    resampled = resampled.stack(level=1).swaplevel(0, 1).sort_index(level=0)
    
    resampled.reset_index(inplace=True)
    resampled = resampled.rename(columns={"level_1": "hour"})
    
    _log(level, "Writing resampled CSV to: %s", output_file_path)
    resampled.to_csv(output_file_path)


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

    # 读取第一个CSV文件并初始化交集集合
    first_file_path = os.path.join(directory_path, csv_files[0])
    first_df = pd.read_csv(first_file_path)
    intersection_set = set(first_df['stay_id'])

    # 遍历其余的CSV文件
    for file in csv_files[1:]:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        intersection_set = intersection_set.intersection(set(df['stay_id']))

    # 将交集集合转换为DataFrame
    aki_id_df = pd.DataFrame(list(intersection_set), columns=['stay_id'])

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
    # 读取AKI_ID.csv获取过滤后的stay_id列表
    aki_id_df = pd.read_csv(aki_id_file_path)
    filtered_stay_ids = set(aki_id_df['stay_id'])

    # 初始化一个空的DataFrame用于合并数据
    merged_df = pd.DataFrame()

    # 列出目录下的所有CSV文件
    directory_path = Path(directory_path)
    csv_files = [f for f in directory_path.glob('*.csv')]
    logger.info("Files to be processed: %s", csv_files)

    # 遍历CSV文件
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        
        # 过滤数据，只保留存在于AKI_ID.csv中的stay_id的行
        filtered_df = df[df['stay_id'].isin(filtered_stay_ids)]

        # 将过滤后的数据添加到合并的DataFrame中
        merged_df = pd.concat([merged_df, filtered_df], ignore_index=True)
    
    if 'subject_id' in merged_df.columns:
        merged_df.drop(['subject_id'], axis=1, inplace=True)
    merged_df.rename(columns={'Unnamed: 1': 'time'}, inplace=True)

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