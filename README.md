# Trauma Data Processing Script

A simple, all-in-one Python script that processes trauma activation data from Excel files and matches it with audio file data extracted by an LLM pipeline. Perfect for beginners - just run one command and get results!

## Features

- **Trauma Activation Filtering**: Filters registry data to include only trauma activations (ED_TTA_TYPE01 > 3)
- **Audio File Processing**: Extracts timestamps from audio filenames and loads associated extracted data
- **Intelligent Matching**: Matches registry data with audio data using time windows and multiple criteria:
  - Time proximity (20-minute window, configurable)
  - Sex matching
  - Age matching
  - Mechanism and injury keyword matching
- **Unified Data Creation**: Creates a comprehensive table with registry data, extracted data, and potential matches
- **Manual Review Support**: Generates samples for manual verification of matching accuracy

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- openpyxl >= 3.0.0
- xlrd >= 2.0.0

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python trauma_data_processor_simple.py
```

**That's it!** The script handles everything else automatically.

## Quick Start

**One command to rule them all!** 🚀

```bash
python trauma_data_processor_simple.py
```

That's it! The script will automatically:
1. ✅ Check dependencies
2. ✅ Create sample data (if needed)
3. ✅ Process trauma data
4. ✅ Show results
5. ✅ Save output files

## Sample Data Description

### Excel Files (Registry Data)
- **Location**: `data/xlsx_files/trauma_registry_sample.xlsx`
- **Content**: 20 trauma activation records
- **Columns**: Patient ID, ED_TTA_TYPE01, Patient Arrival Date & Time, Sex, Age, Mechanism, Injuries, Chief Complaint
- **Filtered**: Only includes records with ED_TTA_TYPE01 > 3 (18 records)

### Audio Files (Audio Data)
- **Location**: `data/audio_files/`
- **Content**: 20 audio files (2023-01-15 to 2023-01-16)
- **Format**: `.wav` files with timestamps in filenames
- **JSON Data**: Each audio file has a corresponding `.json` file containing:
  - Transcript text
  - Age
  - Sex
  - Injury mechanism
  - Injuries
  - Activation page

## Matching Results

The script successfully matched all 18 trauma activation records:

- **Total Records**: 18
- **Records with Matches**: 18 (100%)
- **Records without Matches**: 0
- **Average Matches per Record**: 1.00

### Matching Examples
- **P001**: 25-year-old male, MVA, arrival time 14:25:30 → matched audio 14:30:25 (time diff 4.9 minutes, match score 0.883)
- **P002**: 32-year-old female, fall from height, arrival time 15:10:45 → matched audio 15:15:30 (time diff 4.8 minutes, match score 0.886)

## Usage

### Simple Usage (Recommended)

```bash
python trauma_data_processor_simple.py
```

### Advanced Usage (For Customization)

If you need to customize the processing, you can modify the configuration at the top of `trauma_data_processor_simple.py`:

```python
# Configuration settings
TIME_WINDOW_MINUTES = 20  # Time window for matching
XLSX_FOLDER = "data/xlsx_files"  # Excel files path
AUDIO_FOLDER = "data/audio_files"  # Audio files path
OUTPUT_FOLDER = "output"  # Output path
```

## Data Structure Requirements

### Excel Files (Registry Data)
- Must contain column `ED_TTA_TYPE01` with activation levels
- Must contain column `Patient Arrival Date & Time` (or similar) with arrival timestamps
- Should contain columns for `Sex`, `Age`, `Mechanism`, `Injuries` for matching

### Audio Files
- Audio files should have timestamps in filenames (e.g., `2023-01-15_14-30-25.wav`)
- Associated JSON files should contain extracted data:
  ```json
  {
    "transcript": "Enhanced generated transcript",
    "age": 25,
    "sex": "M",
    "mechanism": "Motor vehicle accident",
    "injuries": "Head injury, chest trauma",
    "activation_page": "LLM generated activation page"
  }
  ```


## Matching Algorithm

The matching algorithm uses a scoring system based on:

1. **Time Proximity** (30% weight): Closer timestamps score higher
2. **Sex Matching** (30% weight): Exact sex matches get full points
3. **Age Matching** (20% weight): Age differences ≤5 years get full points, ≤10 years get partial points
4. **Mechanism Matching** (10% weight): Keyword-based similarity using Jaccard index
5. **Injury Matching** (10% weight): Keyword-based similarity using Jaccard index

## Configuration Options

- **Time Window**: Default 20 minutes, configurable via `processor.time_window`
- **Manual Sample Size**: Default 100 records, configurable in `run_full_pipeline()`
- **Output Format**: Both Excel (.xlsx) and CSV formats available

## Output Files

After processing, you'll find these files in the `output/` folder:

1. **`trauma_data_registry_filtered.xlsx`** - Filtered trauma activation data
2. **`trauma_data_audio_data.xlsx`** - Processed audio data
3. **`trauma_data_unified.xlsx`** - Unified data table with matching results
4. **`trauma_data_unified.csv`** - CSV format of unified data
5. **`manual_checking_sample.xlsx`** - Sample for manual review (10 records)

## Customization

You can modify the configuration at the top of `trauma_data_processor_simple.py`:

- **Time Window**: `TIME_WINDOW_MINUTES = 20` (matching time range)
- **Matching Weights**: Time proximity, sex matching, age matching, etc.
- **File Paths**: Locations of Excel and audio files

## Scaling to Full Dataset

When you're ready to process your full dataset:

1. Place your Excel files in `data/xlsx_files/`
2. Place audio files and JSON data in `data/audio_files/`
3. Adjust the `manual_sample_size` parameter
4. Run the complete processing pipeline

## File Structure

```
ems_matching/
├── trauma_data_processor_simple.py  # Main script (one file does it all!)
├── requirements.txt                 # Python dependencies
├── README.md                       # Complete documentation
├── SIMPLE_USAGE.md                 # Simple usage guide
├── PROJECT_OVERVIEW.md             # Project overview
├── data/                           # Sample data
│   ├── xlsx_files/                # Excel files with registry data
│   └── audio_files/               # Audio files and extracted JSON data
└── output/                         # Output files (auto-generated)
    ├── trauma_data_unified.xlsx    # Main results
    ├── trauma_data_unified.csv     # CSV version
    └── other output files...
```


