# Drilling MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server for oil and gas drilling data analysis. 

This is a mockup server intended to provides tools, resources, and prompt templates for analyzing drilling data from CSV files, with support for Rate of Penetration (ROP), Mechanical Specific Energy (MSE), Non-Productive Time (NPT) calculations, and data visualization.

## Features

### 📊 Resources
- **`drilling://wells`** - List all available wells and their file types (time/depth)
- **`drilling://well/{well_name}/{file_type}`** - Get detailed information about a specific well's data

### 🔧 Tools
- **`list_wells`** - List all wells with metadata (converted from resource for tool access)
- **`inspect_headers`** - Inspect column headers of a CSV file
- **`calculate_rop`** - Calculate Rate of Penetration statistics with filtering
- **`calculate_mse`** - Calculate Mechanical Specific Energy (MSE) for drilling efficiency
- **`calculate_npt`** - Calculate Non-Productive Time (NPT) analysis
- **`plot_data`** - Create horizontal time-based plots with resampling support
- **`filter_data`** - Filter well data by time or depth windows

### 📝 Prompt Templates
- **`list_avialable_wells`** - Generate a prompt to list available wells
- **`analyze_rop_performance`** - Comprehensive ROP performance analysis workflow
- **`analyze_drilling_efficiency`** - Full drilling efficiency analysis
- **`compare_wells`** - Compare multiple wells' performance
- **`optimize_drilling_parameters`** - Parameter optimization analysis
- **`analyze_custom_file_rop`** - Analyze ROP from custom CSV files

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install pandas numpy matplotlib mcp fastmcp nest-asyncio
```

Or use a `requirements.txt` file:

```

Then install:

```bash
pip install -r requirements.txt
```
Or the `setup_conda_env.sh`


## Configuration

### 1. Data Directory Setup

Place your drilling data CSV files in the data directory. Files should be named with the pattern:
- `{well_name} time.csv` for time-based data
- `{well_name} depth.csv` for depth-based data

**Default location**: `data/drilling/` (relative to the server script)

**Custom location**: Set the `DRILLING_DATA_DIR` environment variable:

```bash
# Linux/macOS
export DRILLING_DATA_DIR="/path/to/your/drilling/data"

# Windows
set DRILLING_DATA_DIR=C:\path\to\your\drilling\data

# Or in your shell profile (.bashrc, .zshrc, etc.)
export DRILLING_DATA_DIR="$HOME/drilling_data"
```

### 2. MCP Server Configuration

Add the server to your MCP client configuration file (e.g., `server_config.json`):

```json
{
    "mcpServers": {
        "drilling": {
            "command": "python",
            "args": ["drilling_mcp_server.py"],
            "env": {
                "DRILLING_DATA_DIR": "/path/to/your/data"
            }
        }
    }
}
```

**Note**: Make sure to use the full path to `drilling_mcp_server.py` or ensure it's in your PATH.

## Usage

### Starting the Server

```bash
python drilling_mcp_server.py
```

The server runs on stdio transport and communicates via the MCP protocol.

### Using with MCP Clients

#### Claude Desktop

1. Add the server configuration to your Claude Desktop config file (Claude> Settings > Developer > Edit Config)
2. In Claude desktop add
```
"drilling": {
          "command": "LOCAL_PATH_TO_YOUR_PYTHON",
          "args": [
            "LOCAL_PATH_TO/drilling_mcp_server.py" 
          ]
      },
```
Make sure to use the right python, activate your python env and run `which python` to have the full path of your local python

2. Restart Claude Desktop

3. The drilling tools and resources will be available in your conversations

#### Programmatic Usage

See `mcp_chatbot.py` for an example of how to connect to and use the server programmatically.

Run 
```bash
python mcp_chatbot.py
```

### Resource Access

Access resources using the `@` syntax in MCP clients:

```
@drilling://wells
@drilling://well/Norway-NA-15_$47$_9-F-1/time
```

### Tool Usage Examples

#### Calculate ROP Statistics

```python
calculate_rop(
    well_name="Norway-NA-15_$47$_9-F-1",
    file_type="time",
    start_time="2007-12-01T00:00:00Z",
    end_time="2007-12-02T00:00:00Z"
)
```

#### Calculate MSE

```python
calculate_mse(
    well_name="Norway-NA-15_$47$_9-F-1",
    file_type="time",
    min_depth=100.0,
    max_depth=500.0
)
```

#### Plot Data

```python
plot_data(
    well_name="Norway-NA-15_$47$_9-F-1",
    columns=["Rate of Penetration m/h", "Weight on Bit kkgf"],
    interval="30s",
    output_format="json"
)
```

#### Filter Data

```python
filter_data(
    well_name="Norway-NA-15_$47$_9-F-1",
    file_type="time",
    start_time="2007-12-01T00:00:00Z",
    end_time="2007-12-02T00:00:00Z",
    output_file="filtered_data.csv"
)
```

### Custom File Support

All tools support custom input files via the `custom_file` parameter:

```python
calculate_rop(
    well_name="custom_well",
    file_type="time",
    custom_file="/path/to/your/data.csv"
)
```

**Note**: Custom files must be under 5MB by default (configurable via `MAX_FILE_SIZE_MB`).

## License

This project is part of a learning exercise for MCP server development. Please check the repository for license information.

## Acknowledgments
- Sample Drilling data were extracted from https://www.ux.uis.no/~atunkiel/file_list.html 
which uses Volve dataset published by Equinor under Creative Commons (CC BY-NC-SA 4.0) license.
- Built with [FastMCP](https://github.com/jlowin/fastmcp)
- Uses the [Model Context Protocol](https://modelcontextprotocol.io/)
- Data preprocessing utilities for efficient handling of large drilling datasets

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is a Mockup server is designed for analysis of drilling data. Ensure you have proper authorization to use any drilling datasets and comply with data usage agreements.
