import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from datetime import datetime
import zipfile
import io
from functools import lru_cache
from mcp.server.fastmcp import FastMCP
from data_preprocessing import plot_horizontal_time_curves
import base64

# Initialize FastMCP server
mcp = FastMCP("drilling")

# Data directory - configurable via environment variable for deployment
# Falls back to script-relative path if not set
_script_dir = Path(__file__).parent.absolute()
_data_dir_env = os.getenv("DRILLING_DATA_DIR")
if _data_dir_env:
    # Use environment variable if set (allows deployment configuration)
    DATA_DIR = Path(_data_dir_env).absolute()
else:
    # Default: use relative path based on script location
    # This ensures the path works regardless of where the server is run from
    DATA_DIR = _script_dir / "data" / "drilling"

# Configuration for large file handling
MAX_FILE_SIZE_MB = 5  # Files larger than this will be processed in chunks
CHUNK_SIZE = 10000  # Number of rows per chunk
MAX_CACHE_SIZE_MB = 500  # Maximum memory for caching
SAMPLE_SIZE = 1_000_000  # Default sample size for large files

# Simple in-memory cache with size tracking
_data_cache: Dict[str, pd.DataFrame] = {}
_cache_sizes: Dict[str, float] = {}  # Size in MB

def get_well_files():
    """Get all available well files from the data directory."""
    if not DATA_DIR.exists():
        # Provide helpful error message for debugging
        print(f"WARNING: Data directory not found: {DATA_DIR}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {_script_dir}")
        if not _data_dir_env:
            print(f"Tip: Set DRILLING_DATA_DIR environment variable to specify a custom data directory")
        return []
    
    files = []
    for file in DATA_DIR.iterdir():
        if file.suffix == '.csv':
            # Extract well name and type (time/depth)
            name = file.stem
            if ' time' in name.lower():
                well_name = name.replace(' time', '').replace(' Time', '')
                file_type = 'time'
            elif ' depth' in name.lower():
                well_name = name.replace(' depth', '').replace(' Depth', '')
                file_type = 'depth'
            else:
                continue
            
            files.append({
                'well_name': well_name,
                'file_type': file_type,
                'file_path': str(file),
                'filename': file.name
            })
    
    return files

def check_file_size(file_path: str, size_limit_mb: int = 5) -> float:
    """
    Check the size of the file is under a certain size limit of size_limit in MB.
    Throw an error if the file exceeds the size limit. Does nothing otherwise.
    """
    
    MAX_SIZE_BYTES = 5 * 1024 * 1024 # 5MB
    # Get the file size in bytes
    file_size = os.path.getsize(file_path)
    # Convert the file size to MB
    file_size_mb = file_size / 1024 / 1024

    if file_size_mb > MAX_SIZE_BYTES:
        raise ValueError(f"File limit exceeded. Input data must be under {size_limit_mb} MB. Current size: {file_size_mb:.2f} MB")

def load_well_data(well_name: str, file_type: str = 'time', custom_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load well data from CSV file.
    
    Args:
        well_name: Name of the well
        file_type: 'time' or 'depth'
        custom_file: Optional path to custom CSV file
        
    Returns:
        DataFrame with well data
    """
    if custom_file:
        if not os.path.exists(custom_file):
            raise FileNotFoundError(f"Custom file not found: {custom_file}")
        
        check_file_size(custom_file, size_limit_mb=MAX_FILE_SIZE_MB)

        return pd.read_csv(custom_file, nrows=SAMPLE_SIZE, low_memory=False)
    
    # Find the file in data directory
    files = get_well_files()
    
    matching_file = None
    
    for file_info in files:
        if file_info['well_name'] == well_name and file_info['file_type'] == file_type:
            matching_file = file_info['file_path']
            break
    
    if not matching_file:
        raise FileNotFoundError(f"No {file_type} file found for well: {well_name}")
    
    return pd.read_csv(matching_file, nrows=SAMPLE_SIZE, low_memory=False)

def filter_by_time_window(df: pd.DataFrame, start_time: Optional[str] = None, 
                          end_time: Optional[str] = None) -> pd.DataFrame:
    """Filter dataframe by time window."""
    if 'DateTime parsed' in df.columns:
        df['DateTime parsed'] = pd.to_datetime(df['DateTime parsed'], errors='coerce')
        if start_time:
            df = df[df['DateTime parsed'] >= pd.to_datetime(start_time)]
        if end_time:
            df = df[df['DateTime parsed'] <= pd.to_datetime(end_time)]
    elif 'Time s' in df.columns:
        if start_time:
            start_seconds = float(start_time) if start_time.replace('.', '').isdigit() else None
            if start_seconds is not None:
                df = df[df['Time s'] >= start_seconds]
        if end_time:
            end_seconds = float(end_time) if end_time.replace('.', '').isdigit() else None
            if end_seconds is not None:
                df = df[df['Time s'] <= end_seconds]
    
    return df

def filter_by_depth_window(df: pd.DataFrame, min_depth: Optional[float] = None, 
                           max_depth: Optional[float] = None) -> pd.DataFrame:
    """Filter dataframe by depth window."""
    depth_cols = ['Bit Depth (MD) m', 'Measured Depth m', 'Hole depth (MD) m', 'Bit Depth m']
    depth_col = None
    
    for col in depth_cols:
        if col in df.columns:
            depth_col = col
            break
    
    if depth_col:
        if min_depth is not None:
            df = df[df[depth_col] >= min_depth]
        if max_depth is not None:
            df = df[df[depth_col] <= max_depth]
    
    return df

# ==================== RESOURCES ====================
@mcp.resource("drilling://wells")
def list_wells() -> str:
    """
    List all wells and their available files (time/depth).

    Returns:
        JSON string containing well metadata, file listings, and summary stats.
    """
    files = get_well_files()
    
    if not files:
        return json.dumps({
            "error": "No well data files found",
            "data_directory": str(DATA_DIR),
            "directory_exists": DATA_DIR.exists()
        }, indent=2)
    
    wells: Dict[str, Dict[str, Any]] = {}
    for file_info in files:
        well_name = file_info["well_name"]
        entry = wells.setdefault(well_name, {
            "well_name": well_name,
            "has_time": False,
            "has_depth": False,
            "files": []
        })
        
        entry["files"].append({
            "filename": file_info["filename"],
            "file_type": file_info["file_type"],
            "path": file_info["file_path"]
        })
        
        if file_info["file_type"] == "time":
            entry["has_time"] = True
        elif file_info["file_type"] == "depth":
            entry["has_depth"] = True
    
    sorted_wells = sorted(wells.values(), key=lambda item: item["well_name"])
    
    result = {
        "data_directory": str(DATA_DIR),
        "total_wells": len(sorted_wells),
        "total_files": len(files),
        "wells": sorted_wells
    }
    
    return json.dumps(result, indent=2)

@mcp.resource("drilling://well/{well_name}/{file_type}")
def get_well_data(well_name: str, file_type: str) -> str:
    """Get summary information about a specific well's data."""
    try:
        df = load_well_data(well_name, file_type)

        content = f"# Well Data: {well_name} ({file_type})\n\n"
        content += f"**Total Records**: {len(df)}\n\n"
        content += f"**Columns**: {len(df.columns)}\n\n"
        
        # Show key columns if available
        key_columns = []
        if file_type == 'time':
            key_columns = ['DateTime parsed', 'Time s', 'Rate of Penetration m/h', 
                          'Weight on Bit kkgf', 'Average Rotary Speed rpm', 
                          'Average Surface Torque kN.m', 'Bit Depth (MD) m']
        else:
            key_columns = ['Measured Depth m', 'Rate of Penetration m/h', 
                          'Weight on Bit kkgf', 'Average Rotary Speed rpm',
                          'Average Surface Torque kN.m']
        
        content += "## Key Columns Available:\n"
        for col in key_columns:
            if col in df.columns:
                content += f"- ✓ {col}\n"
            else:
                content += f"- ✗ {col} (not available)\n"
        
        content += "\n## Data Range:\n"
        if file_type == 'time' and 'DateTime parsed' in df.columns:
            df['DateTime parsed'] = pd.to_datetime(df['DateTime parsed'], errors='coerce')
            valid_dates = df['DateTime parsed'].dropna()
            if len(valid_dates) > 0:
                content += f"- Start: {valid_dates.min()}\n"
                content += f"- End: {valid_dates.max()}\n"
        elif 'Measured Depth m' in df.columns:
            depth_col = df['Measured Depth m'].dropna()
            if len(depth_col) > 0:
                content += f"- Min Depth: {depth_col.min():.2f} m\n"
                content += f"- Max Depth: {depth_col.max():.2f} m\n"
        
        return content
    except Exception as e:
        return f"# Error\n\nFailed to load well data: {str(e)}"

# ==================== TOOLS ====================
@mcp.tool()
def inspect_headers(file_path="well_A_run1.csv"):
    """ 
    List the column headers of a file.
    Each column represent a drilling feature that was measured at a given time or depth.
    This function helps identify the columns that are available for analysis and index columns.
    Columns are listed in the order they are in the file.

    Args:
        file_path: Path to the file to inspect
    Returns:
        JSON string containing the column headers
    """
    df = pd.read_csv(file_path, nrows=25, low_memory=False)

    return json.dumps(df.columns.tolist(), indent=2)

@mcp.tool()
def calculate_rop(well_name: str, file_type: str = 'time', 
                  start_time: Optional[str] = None, end_time: Optional[str] = None,
                  min_depth: Optional[float] = None, max_depth: Optional[float] = None,
                  custom_file: Optional[str] = None) -> str:
    """
    Calculate Rate of Penetration (ROP) statistics for a well.
    
    Args:
        well_name: Name of the well
        file_type: 'time' or 'depth' (default: 'time')
        start_time: Start time for filtering (ISO format or seconds)
        end_time: End time for filtering (ISO format or seconds)
        min_depth: Minimum depth for filtering (meters)
        max_depth: Maximum depth for filtering (meters)
        custom_file: Optional path to custom CSV file
        
    Returns:
        JSON string with ROP statistics
    """
    try:
        df = load_well_data(well_name, file_type, custom_file)
        
        # Apply filters
        if file_type == 'time':
            df = filter_by_time_window(df, start_time, end_time)
        if min_depth is not None or max_depth is not None:
            df = filter_by_depth_window(df, min_depth, max_depth)
        
        # Find ROP column
        rop_cols = ['Rate of Penetration m/h', 'Rate of penetration m/h', 
                   'Rate of Penetration (5ft avg) ', 'ROP m/h']
        rop_col = None
        for col in rop_cols:
            if col in df.columns:
                rop_col = col
                break
        
        if rop_col is None:
            return json.dumps({"error": "ROP column not found in data"}, indent=2)
        
        rop_data = df[rop_col].dropna()
        
        if len(rop_data) == 0:
            return json.dumps({"error": "No valid ROP data after filtering"}, indent=2)
        
        stats = {
            "well_name": well_name,
            "file_type": file_type,
            "rop_column": rop_col,
            "statistics": {
                "mean": float(rop_data.mean()),
                "median": float(rop_data.median()),
                "std": float(rop_data.std()),
                "min": float(rop_data.min()),
                "max": float(rop_data.max()),
                "count": int(len(rop_data))
            },
            "filters_applied": {
                "start_time": start_time,
                "end_time": end_time,
                "min_depth": min_depth,
                "max_depth": max_depth
            }
        }
        
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
def calculate_mse(well_name: str, file_type: str = 'time',
                  start_time: Optional[str] = None, end_time: Optional[str] = None,
                  min_depth: Optional[float] = None, max_depth: Optional[float] = None,
                  custom_file: Optional[str] = None) -> str:
    """
    Calculate Mechanical Specific Energy (MSE) for a well.
    MSE = (WOB/A) + (120π*RPM*T)/(A*ROP)
    where WOB = Weight on Bit, A = bit area, RPM = rotary speed, T = torque, ROP = rate of penetration
    
    Args:
        well_name: Name of the well
        file_type: 'time' or 'depth' (default: 'time')
        start_time: Start time for filtering
        end_time: End time for filtering
        min_depth: Minimum depth for filtering (meters)
        max_depth: Maximum depth for filtering (meters)
        custom_file: Optional path to custom CSV file
        
    Returns:
        JSON string with MSE statistics
    """
    try:
        df = load_well_data(well_name, file_type, custom_file)
        
        # Apply filters
        if file_type == 'time':
            df = filter_by_time_window(df, start_time, end_time)
        if min_depth is not None or max_depth is not None:
            df = filter_by_depth_window(df, min_depth, max_depth)
        
        # Find required columns
        wob_col = None
        rpm_col = None
        torque_col = None
        rop_col = None
        
        for col in df.columns:
            if 'Weight on Bit' in col or 'WOB' in col:
                wob_col = col
            if 'Rotary Speed' in col or 'RPM' in col:
                rpm_col = col
            if 'Torque' in col:
                torque_col = col
            if 'Rate of Penetration' in col or 'ROP' in col:
                rop_col = col
        
        if not all([wob_col, rpm_col, torque_col, rop_col]):
            missing = [name for name, col in [
                ('WOB', wob_col), ('RPM', rpm_col), 
                ('Torque', torque_col), ('ROP', rop_col)
            ] if col is None]
            return json.dumps({
                "error": f"Missing required columns: {', '.join(missing)}"
            }, indent=2)
        
        # Assume bit diameter (need to estimate or use default)
        # Common bit sizes: 8.5" = 0.216 m, 12.25" = 0.311 m
        bit_diameter = 0.216  # Default 8.5 inches in meters
        bit_area = np.pi * (bit_diameter / 2) ** 2
        
        # Calculate MSE
        wob = pd.to_numeric(df[wob_col], errors='coerce')
        rpm = pd.to_numeric(df[rpm_col], errors='coerce')
        torque = pd.to_numeric(df[torque_col], errors='coerce')
        rop = pd.to_numeric(df[rop_col], errors='coerce')
        
        # Convert units if needed
        # WOB might be in kkgf (kilogram-force), convert to N
        if 'kkgf' in wob_col:
            wob = wob * 9.80665 * 1000  # Convert to Newtons
        
        # Torque might be in kN.m, convert to N.m
        if 'kN.m' in torque_col:
            torque = torque * 1000  # Convert to N.m
        
        # ROP in m/h, convert to m/s
        rop_ms = rop / 3600
        
        # Calculate MSE: (WOB/A) + (120π*RPM*T)/(A*ROP)
        # Avoid division by zero
        valid_mask = (wob.notna() & rpm.notna() & torque.notna() & 
                     rop.notna() & (rop_ms > 0))
        
        mse = np.zeros(len(df))
        mse[valid_mask] = (
            (wob[valid_mask] / bit_area) + 
            (120 * np.pi * rpm[valid_mask] * torque[valid_mask]) / 
            (bit_area * rop_ms[valid_mask])
        )
        
        mse_series = pd.Series(mse, index=df.index)
        mse_series = mse_series[mse_series > 0]  # Remove invalid values
        
        if len(mse_series) == 0:
            return json.dumps({"error": "No valid MSE data calculated"}, indent=2)
        
        stats = {
            "well_name": well_name,
            "file_type": file_type,
            "mse_statistics": {
                "mean": float(mse_series.mean()),
                "median": float(mse_series.median()),
                "std": float(mse_series.std()),
                "min": float(mse_series.min()),
                "max": float(mse_series.max()),
                "count": int(len(mse_series)),
                "unit": "Pa (Pascals)"
            },
            "assumptions": {
                "bit_diameter_m": bit_diameter,
                "bit_area_m2": bit_area
            },
            "filters_applied": {
                "start_time": start_time,
                "end_time": end_time,
                "min_depth": min_depth,
                "max_depth": max_depth
            }
        }
        
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
def calculate_npt(well_name: str, file_type: str = 'time',
                  start_time: Optional[str] = None, end_time: Optional[str] = None,
                  custom_file: Optional[str] = None) -> str:
    """
    Calculate Non-Productive Time (NPT) for a well.
    NPT is typically identified by rig mode changes or periods where ROP is zero.
    
    Args:
        well_name: Name of the well
        file_type: 'time' or 'depth' (default: 'time')
        start_time: Start time for filtering
        end_time: End time for filtering
        custom_file: Optional path to custom CSV file
        
    Returns:
        JSON string with NPT statistics
    """
    try:
        df = load_well_data(well_name, file_type, custom_file)
        
        # Apply time filter
        if file_type == 'time':
            df = filter_by_time_window(df, start_time, end_time)
        
        # Find rig mode column
        rig_mode_col = None
        for col in df.columns:
            if 'Rig Mode' in col or 'rig mode' in col.lower():
                rig_mode_col = col
                break
        
        # Find ROP column
        rop_col = None
        for col in df.columns:
            if 'Rate of Penetration' in col or 'ROP' in col:
                rop_col = col
                break
        
        npt_periods = []
        total_npt_hours = 0
        
        if rig_mode_col:
            # Identify non-drilling modes
            non_drilling_modes = ['Tripping', 'Connection', 'Off Bottom', 'Standing', 
                                 'Circulating', 'Conditioning', 'Wiper Trip']
            
            df['is_drilling'] = ~df[rig_mode_col].astype(str).str.contains(
                '|'.join(non_drilling_modes), case=False, na=False
            )
            
            # Find time column
            time_col = None
            if 'DateTime parsed' in df.columns:
                time_col = 'DateTime parsed'
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            elif 'Time s' in df.columns:
                time_col = 'Time s'
            
            if time_col:
                # Group consecutive non-drilling periods
                df['group'] = (df['is_drilling'] != df['is_drilling'].shift()).cumsum()
                
                for group_id, group_df in df.groupby('group'):
                    if not group_df['is_drilling'].iloc[0]:
                        if time_col == 'DateTime parsed':
                            duration = (group_df[time_col].max() - group_df[time_col].min()).total_seconds() / 3600
                            start = group_df[time_col].min()
                            end = group_df[time_col].max()
                        else:
                            duration = (group_df[time_col].max() - group_df[time_col].min()) / 3600
                            start = group_df[time_col].min()
                            end = group_df[time_col].max()
                        
                        npt_periods.append({
                            "start": str(start),
                            "end": str(end),
                            "duration_hours": float(duration),
                            "rig_mode": group_df[rig_mode_col].iloc[0] if rig_mode_col else "Unknown"
                        })
                        total_npt_hours += duration
        
        # Alternative: Use ROP = 0 as NPT indicator
        if rop_col and len(npt_periods) == 0:
            rop = pd.to_numeric(df[rop_col], errors='coerce')
            zero_rop_mask = (rop == 0) | rop.isna()
            
            if time_col:
                if time_col == 'DateTime parsed':
                    total_npt_hours = (df[zero_rop_mask][time_col].max() - 
                                      df[zero_rop_mask][time_col].min()).total_seconds() / 3600
                else:
                    total_npt_hours = (df[zero_rop_mask][time_col].max() - 
                                      df[zero_rop_mask][time_col].min()) / 3600
        
        # Calculate total time
        total_time_hours = 0
        if time_col:
            if time_col == 'DateTime parsed':
                total_time_hours = (df[time_col].max() - df[time_col].min()).total_seconds() / 3600
            else:
                total_time_hours = (df[time_col].max() - df[time_col].min()) / 3600
        
        npt_percentage = (total_npt_hours / total_time_hours * 100) if total_time_hours > 0 else 0
        
        result = {
            "well_name": well_name,
            "file_type": file_type,
            "npt_statistics": {
                "total_npt_hours": float(total_npt_hours),
                "total_time_hours": float(total_time_hours),
                "npt_percentage": float(npt_percentage),
                "npt_periods_count": len(npt_periods)
            },
            "npt_periods": npt_periods[:10],  # Limit to first 10 periods
            "filters_applied": {
                "start_time": start_time,
                "end_time": end_time
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
def plot_data(well_name: str, columns: Optional[List[str]], output_dir: str = DATA_DIR,
              start_time: Optional[str] = None, end_time: Optional[str] = None,
              min_depth: Optional[float] = None, max_depth: Optional[float] = None,
              custom_file: Optional[str] = None, 
              interval: str = "30s", output_format: Optional[str] = "json") -> str:
    """
    Plot drilling data using horizontal time curves layout (time-based data only).
    
    Args:
        well_name: Name of the well
        columns: list of columns to plot
        start_time: Start time for filtering
        end_time: End time for filtering
        custom_file: Optional path to custom CSV file
        output_dir: Optional path to save the plot. If None, generates default filename.
        interval: Resampling interval for time-based data (default: '30s')

        
    Returns:
        JSON string with plot information including saved figure path
    """
    try:
        df = load_well_data(well_name, "time", custom_file)
        
        # Apply filters
        df = filter_by_time_window(df, start_time, end_time)
        
        # Check if columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return json.dumps({"error": f"Columns not found in data: {', '.join(missing_cols)}"}, indent=2)
        
        # Find datetime column
        datetime_col = None
        for col in ['DateTime parsed', 'Time s', 'Time Time']:
            if col in df.columns or (hasattr(df.index, 'name') and df.index.name == col):
                datetime_col = col
                break
        
        # Generate default output path if not provided
        safe_well_name = well_name.replace(' ', '_').replace('/', '_')
        output_path = os.path.join(output_dir, f"plot_{safe_well_name}.png")
        
        # Use the preprocessing function - always save the figure
        result = plot_horizontal_time_curves(
            df=df,
            columns=columns,
            index_column=datetime_col,
            interval=interval,
            output_path=output_path
        )
        
            
        output_img = result.get('output_path', output_path)
        columns = result.get('columns', columns)
        interval = result.get('interval', interval)
        data_points = result.get('points', len(df))
        image_data = ''

        with open(output_img, 'rb') as image_file:
            # Read the image file and decode it to base64
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
        if output_format == "media":
            return {
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Plot created with {len(df)} data points at {interval} intervals"
                    }
                ]
            }
            # Return base64 image data
            #return {
                #"type": "image",
                #"data": image_data,
                #"mimeType": "image/png",
                #"metadata": {
                #    "columns": columns,
                #    "interval": interval,
                #    "data_points": len(df),
                #    "well_name": well_name
                #}
            #}
        
        elif output_format == "json":
            # For JSON format, include the base64 data as well
            return json.dumps({
                "success": True,
                "plot_path": output_path,  # Keep for reference
                "image_data": image_data,   # Add base64 data
                "mimeType": "image/png",
                "columns": columns,
                "interval": interval,
                "data_points": len(df),
                "figure_saved": True
            }, indent=2)
            
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
def filter_data(well_name: str, file_type: str = 'time',
                start_time: Optional[str] = None, end_time: Optional[str] = None,
                min_depth: Optional[float] = None, max_depth: Optional[float] = None,
                custom_file: Optional[str] = None, output_file: Optional[str] = None) -> str:
    """
    Filter well data by time or depth window and optionally save to file.
    
    Args:
        well_name: Name of the well
        file_type: 'time' or 'depth' (default: 'time')
        start_time: Start time for filtering
        end_time: End time for filtering
        min_depth: Minimum depth for filtering (meters)
        max_depth: Maximum depth for filtering (meters)
        custom_file: Optional path to custom CSV file
        output_file: Optional path to save filtered data (CSV format)
        
    Returns:
        JSON string with filtering results
    """
    try:
        df = load_well_data(well_name, file_type, custom_file)
        original_count = len(df)
        
        # Apply filters
        if file_type == 'time':
            df = filter_by_time_window(df, start_time, end_time)
        if min_depth is not None or max_depth is not None:
            df = filter_by_depth_window(df, min_depth, max_depth)
        
        filtered_count = len(df)
        
        # Save if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
        
        result = {
            "well_name": well_name,
            "file_type": file_type,
            "filtering_results": {
                "original_records": original_count,
                "filtered_records": filtered_count,
                "records_removed": original_count - filtered_count,
                "retention_percentage": (filtered_count / original_count * 100) if original_count > 0 else 0
            },
            "filters_applied": {
                "start_time": start_time,
                "end_time": end_time,
                "min_depth": min_depth,
                "max_depth": max_depth
            }
        }
        
        if output_file:
            result["output_file"] = output_file
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

# ==================== PROMPTS ====================
@mcp.prompt()
def list_avialable_wells() -> str:
    """List all available wells in the data directory."""
    return """Call the `list_wells` tool with no arguments to retrieve the well inventory.

After calling the tool:
- Produce a table with columns: Well Name, File Name(s), Time Data (Yes/No), Depth Data (Yes/No)
- For wells with multiple files, join filenames with commas in the File Name(s) column
- Indicate “Yes” when any file of that type exists for the well, otherwise “No”
- Do not perform extra calculations or commentary beyond presenting the table"""
    

@mcp.prompt()
def analyze_rop_performance(well_name: str, file_type: str = 'time') -> str:
    """Generate a prompt to analyze ROP performance for a well."""
    return f"""Analyze the Rate of Penetration (ROP) performance for well '{well_name}' using {file_type} data.

Please:
1. Calculate ROP statistics using the calculate_rop tool
2. Identify periods of high and low ROP
3. Plot ROP versus time or depth using the plot_data tool
4. Compare ROP performance across different depth intervals
5. Provide recommendations for optimizing drilling performance

Use the available tools to:
- calculate_rop: Get ROP statistics
- plot_data: Visualize ROP trends
- filter_data: Focus on specific time or depth windows"""

@mcp.prompt()
def analyze_drilling_efficiency(well_name: str, file_type: str = 'time') -> str:
    """Generate a prompt to analyze overall drilling efficiency."""
    return f"""Perform a comprehensive drilling efficiency analysis for well '{well_name}' using {file_type} data.

Please:
1. Calculate ROP statistics using calculate_rop
2. Calculate MSE (Mechanical Specific Energy) using calculate_mse
3. Calculate NPT (Non-Productive Time) using calculate_npt
4. Create plots showing:
   - ROP vs depth/time
   - MSE vs depth/time
   - WOB vs RPM
5. Identify correlations between drilling parameters
6. Provide insights on:
   - Optimal drilling parameters
   - Areas of inefficiency
   - Recommendations for improvement

Use all available tools to gather comprehensive insights."""

@mcp.prompt()
def compare_wells(well_names: str) -> str:
    """Generate a prompt to compare multiple wells."""
    well_list = [w.strip() for w in well_names.split(',')]
    return f"""Compare drilling performance across multiple wells: {', '.join(well_list)}.

Please:
1. Calculate ROP statistics for each well
2. Calculate MSE for each well
3. Calculate NPT for each well
4. Create comparative plots showing:
   - ROP comparison
   - MSE comparison
   - NPT comparison
5. Identify best and worst performing wells
6. Analyze factors contributing to performance differences
7. Provide recommendations based on best practices from top performers

Use the calculate_rop, calculate_mse, calculate_npt, and plot_data tools for each well."""

@mcp.prompt()
def optimize_drilling_parameters(well_name: str, file_type: str = 'time') -> str:
    """Generate a prompt to optimize drilling parameters."""
    return f"""Analyze and optimize drilling parameters for well '{well_name}' using {file_type} data.

Please:
1. Calculate ROP and MSE for different parameter ranges
2. Create plots showing relationships between:
   - WOB vs ROP
   - RPM vs ROP
   - Torque vs ROP
   - MSE vs ROP
3. Identify optimal parameter ranges that maximize ROP while minimizing MSE
4. Filter data by depth intervals to analyze parameter optimization at different formations
5. Provide specific recommendations for:
   - Optimal WOB range
   - Optimal RPM range
   - Optimal flow rate (if available)
   - Bit selection considerations

Use calculate_rop, calculate_mse, plot_data, and filter_data tools to perform this analysis."""

@mcp.prompt()
def analyze_custom_file_rop(custom_file_path: str, file_type: str = 'time') -> str:
    """Generate a prompt to analyze ROP from a custom file."""
    return f"""Calculate and analyze Rate of Penetration (ROP) statistics from the custom drilling data file at: {custom_file_path}

Please:
1. Use the calculate_rop tool with custom_file parameter set to '{custom_file_path}' and file_type='{file_type}'
2. Display the ROP statistics (mean, median, std, min, max)
3. Provide a brief interpretation of the results

Use the calculate_rop tool with:
- custom_file: '{custom_file_path}'
- file_type: '{file_type}'
- well_name: can be any placeholder value (e.g., 'custom_well')"""


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')

