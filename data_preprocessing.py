"""
Utilities for preprocessing and resampling drilling CSV files.

This module currently provides helpers to load a lightweight sample of each
well log file (first N lines) and clean the data so that downstream
processing is faster and more reliable.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def ensure_time_index_set(df: pd.DataFrame, index_column: str = "DateTime parsed") -> pd.DataFrame:
    """
    Set dataframe index to 'index_column' if the dataframe is not already with time index, does nothing otherwise.
    Returns the dataframe with the time index set.
    """

    if not pd.api.types.is_datetime64_any_dtype(df.index.dtype):
        if index_column not in df.columns:
            raise ValueError(f"Datetime column '{index_column}' not found in DataFrame")

        df[index_column] = pd.to_datetime(df[index_column], errors="coerce")
        df = df.dropna(subset=[index_column])

        # Set index for resampling
        df = df.set_index(index_column).sort_index()

    return df

def sample_and_clean_csv(
    file_path: str,
    n_rows: int = 1000,
    index_column: str = "DateTime parsed",
    is_time_series: bool = True,
) -> pd.DataFrame:
    """
    Read the first `n_rows` rows of a CSV file and perform cleaning steps:

    1. Drop unnamed/empty-header columns.
    2. Convert the datetime column to pandas datetime.
    3. Drop rows where the datetime conversion failed (NaT/None).
    4. Attempt to convert every other column to numeric, replacing values that
       fail conversion with ``None``.
    5. Drop columns that are entirely empty (all ``None``).

    Args:
        file_path: Absolute or relative path to the CSV file.
        n_rows: Number of initial rows to read (default 1000).
        datetime_column: Name of the datetime column (default "DateTime parsed").

    Returns:
        Cleaned pandas DataFrame containing at most `n_rows` rows.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the specified datetime column is missing.
    """

    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(csv_path, nrows=n_rows, low_memory=False)

    # 1. Remove columns without meaningful headers (Unnamed, empty, NaN)
    unnamed_columns = [
        col
        for col in df.columns
        if pd.isna(col)
        or (isinstance(col, str) and col.strip() == "")
        or str(col).startswith("Unnamed")
    ]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    if index_column not in df.columns:
        raise ValueError(
            f"Expected datetime column '{index_column}' not found in {file_path}"
        )

    # 1 & 2. Convert datetime column and drop rows where conversion fails.
    if is_time_series:
        df[index_column] = pd.to_datetime(df[index_column], errors="coerce")
        df = df.dropna(subset=[index_column])
    else:
        df[index_column] = pd.to_numeric(df[index_column], errors="coerce")
        df = df.dropna(subset=[index_column])

    # 3. Convert remaining columns to numeric when possible.
    for col in df.columns:
        if col == index_column:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        df[col] = converted.where(~converted.isna(), None)

    # 4. Remove columns that are fully empty (all None / NaN).
    df = df.dropna(axis=1, how="all")
    df.set_index(index_column, inplace=True)
    df.sort_index(inplace=True)
    
    return df


def resample_to_interval(
    df: pd.DataFrame,
    datetime_column: str = "DateTime parsed",
    interval: str = "30S",
) -> pd.DataFrame:
    """
    Resample a cleaned DataFrame to a fixed time interval using mean values.

    Args:
        df: Input DataFrame (must already have a valid datetime column).
        datetime_column: Name of the datetime column (default "DateTime parsed").
        interval: Pandas offset alias for resampling interval (default "30S").

    Returns:
        A new DataFrame resampled to the given interval, using mean aggregation
        for all numeric columns.
    """
    df = ensure_time_index_set(df, datetime_column)

    # Convert all remaining columns to numeric where possible for aggregation
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resample using mean
    resampled = df.resample(interval).mean()

    # Drop columns that are fully NaN after resampling
    resampled = resampled.dropna(axis=1, how="all")

    return resampled


def plot_horizontal_time_curves(
    df: pd.DataFrame,
    columns: List[str],
    index_column: str = "DateTime parsed",
    interval: str = "30s",
    output_path: Optional[str] = None,
):
    """
    Plot multiple columns against the same time index using horizontal layout subplots.

    Args:
        df: Input DataFrame containing the data to plot.
        columns: List of column names to plot as x-values.
        datetime_column: Column containing datetime information (default "DateTime parsed").
        interval: Resampling interval (pandas offset alias, default "30S").
        output_path: Optional path to save the plot image. If None, returns the figure.

    Returns:
        Dictionary with plot metadata. Includes "figure" when output_path is None.
    """
    if not columns:
        raise ValueError("At least one column must be provided for plotting.")

    df = ensure_time_index_set(df, index_column)

    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {', '.join(missing)}")

    working_df = df.copy()

    for col in columns:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    if interval:
        working_df = resample_to_interval(
            working_df,
            datetime_column=index_column,
            interval=interval,
        )

    working_df = working_df.dropna(subset=columns, how="all")
    if working_df.empty:
        raise ValueError("No data available after cleaning/resampling for the requested columns.")

    y_values = working_df.index.values
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(columns),
        figsize=(5 * len(columns), 8),
        sharey=True,
        constrained_layout=True,
    )

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(working_df[col], y_values, linewidth=1.2)
        ax.set_xlabel(col)
        ax.grid(alpha=0.3, linestyle="--")
        ax.invert_yaxis()  # earliest times at the top

    axes[0].set_ylabel("Time")
    axes[0].yaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))

    fig.suptitle(f"Time-indexed Curves ({interval} sampling)", fontsize=14)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {
            "output_path": output_path,
            "columns": columns,
            "interval": interval,
            "points": len(working_df),
        }

    return {"figure": fig, "columns": columns, "interval": interval, "points": len(working_df)}

__all__ = ["sample_and_clean_csv", "resample_to_interval", "plot_horizontal_time_curves"]
