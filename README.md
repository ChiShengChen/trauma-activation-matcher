# Trauma Data Processing Script

A simple, all-in-one Python script that processes trauma activation data from Excel files and matches it with audio file data extracted by an LLM pipeline. Perfect for beginners - just run one command and get results!

⚠️ **DISCLAIMER**: This project contains **synthetic/sample data only**. All patient data, audio files, and transcripts are **fictional** and created for demonstration purposes. No real medical data is included.

## Features

- **Trauma Activation Filtering**: Filters registry data to include only trauma activations (ED_TTA_TYPE01 > 3)
- **TXT File Preprocessing**: Automatically processes .txt files and converts them to JSON format for integration
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

**One command to rule them all!** 

```bash
python trauma_data_processor_simple.py
```

That's it! The script will automatically:
1. ✅ Check dependencies
2. ✅ Create sample data (if needed)
3. ✅ Process trauma data
4. ✅ Show results
5. ✅ Save output files

## Data Structure and Format

### Automated Dataset Creation

#### Data Storage
- **Study Files**: Raw study data siloed to separated directory with subdirectories for:
  - Audio in mu-law format
  - Registry Data
- **Working Directory**: Contains preprocessed, post-processed data
  - Directories for:
    - Audio files in .wav format, identified as matching or being analyzed for matching
    - LLM results
    - Corrected transcripts
    - Parsed corrected transcripts
    - Generated pages
    - Tables for analysis

### Data Structure

#### Audio Files
- **Format**: `.wav` audio files with timestamp names (e.g., `240307205523.wav`)
- **Location**: `data/audio_files/`
- **Processing**: Converted from mu-law format for analysis

#### Trauma Registry Data
- **Format**: `.xlsx` format with standardized columns:
  - Trauma Number
  - Patient Arrival Date & Time
  - Age
  - Gender
  - Injury Date
  - Place of Injury
  - TRISS
  - Injury Type
  - Transfer In
  - ED Arrival Date & Time
  - ED Departure Date & Time
  - SBP on Admission
  - Unassisted Resp Rate on Admission
  - GCS on Admission
  - RTS on Admission
  - Discharge Status
  - ISS
  - Injury Text
  - ED_BYPASS_YN
  - **ED_TTA_TYPE01** (key filtering column)
  - INJ_TXT

#### LLM Extracted Discrete Data
- **Filename format**: `240307205523_corrected_result_parsed_result.txt`
- **Content**: Contains LLM logic ending with designated JSON structure:
  ```
  <|end|><|start|>assistant<|channel|>final<|message|>{"agency":"","unit":"","ETA":"on arrival","age":"","sex":"","moi":"","hr":"","rrq":"","sbp":"","dbp":"","end_tidal":"","rr":"","bgl":"","spo2":"","o2":"","injuries":"","ao":"","GCS":"","LOC":"","ac":"","treatment":"","pregnant":"","notes":""}<|return|>
  ```

#### LLM Enhanced Transcripts
- **Filename format**: `240307205523_corrected_resultresult.txt`
- **Content**: Enhanced transcript ending with designated content:
  ```
  ...<|end|><|start|>assistant<|channel|>final<|message|>Good morning, this is [med control] with an entry notification. Good morning. This is [agency]. We're coming in with a [XX]‑year‑old male who has a dislocated hip. He was previously seen at your facility for this injury and underwent surgery. Vitals: heart rate 80, SpO₂ 99% on room air, blood pressure 170/90. He is conscious and alert. We have administered 400 µg fentanyl with little effect and 15 mg ketamine with no effect; he remains in significant pain. ETA to your doors is approximately 15 minutes. Any questions? No further questions. Cleared. Thank you.<|return|>
  ```

### TXT Files (Text Data) - NEW!
- **Location**: `data/txt_files/`
- **Content**: Raw text files with timestamps in filenames
- **Format**: `.txt` files (e.g., `2023-01-15_14-30-25.txt`)
- **Processing**: Automatically converted to JSON format and integrated into the pipeline
- **Parsing**: Intelligent extraction of:
  - Age (from patterns like "25-year-old", "age 25", etc.)
  - Sex (from "male/female", "M/F", etc.)
  - Mechanism (from trauma keywords)
  - Injuries (from injury patterns)
  - Activation page (auto-generated)

## Data Matching Process

### Trauma Activation Filtering
1. **Extract trauma activations** from the xlsx folder where each row event is a trauma activation (registry data)
2. **Filter by activation level**: Column `ED_TTA_TYPE01` contains the activation level
   - Convert to integer, handling string values and spaces
   - Take all values > 3 (trauma activations only)
   - **Activation levels**:
     - 1 = STAT trauma activation
     - 2 = BASIC trauma activation  
     - 3 = Trauma consult
     - 4 = Not a trauma activation

### Audio File Processing
1. **Create audio data table** where each row event is an audio file processed by the LLM Pipeline
2. **Each audio file contains**:
   - A datetime parsed from the audio file name
   - An associated enhanced generated transcript
   - Extracted discrete data parsed from the generated JSON, including:
     - Age and sex
     - Mechanism of injury (MOI)
     - Injuries
   - An LLM generated activation page

### Matching Algorithm
1. **Time window matching**: Match timestamps in audio filenames with `Patient Arrival Date & Time` column
   - Start with 20-minute window on either side (configurable)
2. **Hierarchical matching criteria** (if multiple matches):
   - **Primary**: Choose matching sex between extracted data and registry data
   - **Secondary**: Choose closest age match
   - **Tertiary**: Match by mechanism and injuries
     - Keyword matching
     - Mechanism prioritized over injuries

### Output Generation
1. **Unified table** with:
   - Registry data
   - Extracted data
   - Column with closest several audio matches or likely potential matches
2. **Manual checking sample**: Review set of 100 records to assess matching correctness
   - Review matched audio and closest audio files
   - Important that these are included in the unified table
3. **Scalability**: Scale to entire 2023-2024 dataset and beyond

## Matching Results

The script successfully processes trauma activation records with intelligent matching:

- **Trauma Activations**: Filtered from registry data (ED_TTA_TYPE01 > 3)
- **Audio Files**: Processed with LLM-extracted discrete data
- **Matching**: Time-based with hierarchical criteria (sex, age, mechanism, injuries)
- **Output**: Unified table with multiple potential matches for manual review

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
- **Required columns**:
  - `ED_TTA_TYPE01` - Activation levels (1=STAT, 2=BASIC, 3=Consult, 4=Not trauma)
  - `Patient Arrival Date & Time` - Arrival timestamps for matching
  - `Age` - Patient age for matching
  - `Gender` - Patient gender for matching
- **Additional recommended columns**:
  - Trauma Number, Injury Date, Place of Injury, TRISS, Injury Type
  - Transfer In, ED Arrival/Departure Date & Time
  - SBP on Admission, Unassisted Resp Rate, GCS, RTS
  - Discharge Status, ISS, Injury Text, ED_BYPASS_YN, INJ_TXT

### Audio Files
- **Format**: `.wav` files with timestamp names (e.g., `240307205523.wav`)
- **Processing**: Converted from mu-law format for analysis
- **Associated data**: LLM-processed files with extracted discrete data

### LLM Processed Files
- **Discrete data files**: `{timestamp}_corrected_result_parsed_result.txt`
  - Contains JSON structure with extracted fields:
    - `age`, `sex`, `moi` (mechanism of injury)
    - `injuries`, `hr` (heart rate), `rr` (respiratory rate)
    - `sbp`, `dbp` (blood pressure), `spo2`, `gcs`
    - `treatment`, `notes`, etc.
- **Enhanced transcripts**: `{timestamp}_corrected_resultresult.txt`
  - Contains LLM-enhanced transcript with medical terminology

### TXT Files (NEW!)
- TXT files should have timestamps in filenames (e.g., `2023-01-15_14-30-25.txt`)
- Place them in the `data/txt_files/` directory
- The script will automatically parse and convert them to JSON format
- Example TXT content:
  ```
  This is a 25-year-old male patient involved in a motor vehicle accident. 
  Patient was unconscious at the scene. Head injury with possible skull fracture. 
  Chest trauma with rib fractures. Patient needs immediate trauma activation.
  ```

⚠️ **Note**: The sample data provided is **synthetic** and for demonstration only. Replace with your actual data for real processing.


## Matching Algorithm

The matching algorithm uses a hierarchical approach with time-based filtering:

### Primary Filtering
1. **Time Window**: 20-minute window on either side of `Patient Arrival Date & Time`
   - Configurable time window for matching
   - Only audio files within this window are considered

### Hierarchical Matching (if multiple matches)
1. **Sex Matching** (Primary): Exact match between extracted data and registry data
2. **Age Matching** (Secondary): Closest age match between extracted and registry data
3. **Mechanism Matching** (Tertiary): Keyword-based similarity for mechanism of injury
4. **Injury Matching** (Tertiary): Keyword-based similarity for injury patterns
   - Mechanism prioritized over injuries in case of ties

### Scoring System (for ranking multiple matches)
- **Time Proximity**: Closer timestamps score higher
- **Sex Match**: Exact matches get full points
- **Age Match**: Age differences ≤5 years get full points, ≤10 years get partial points
- **Mechanism Match**: Keyword-based similarity using Jaccard index
- **Injury Match**: Keyword-based similarity using Jaccard index

## Configuration Options

- **Time Window**: Default 20 minutes, configurable via `TIME_WINDOW_MINUTES`
- **Manual Sample Size**: Default 100 records for manual checking
- **Output Format**: Both Excel (.xlsx) and CSV formats available
- **File Paths**: Configurable directories for data storage
  - `XLSX_FOLDER`: Registry data location
  - `AUDIO_FOLDER`: Audio files and LLM results location
  - `TXT_FOLDER`: Raw text files for preprocessing
  - `OUTPUT_FOLDER`: Processed results location

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

### Production Deployment
When ready to process your **actual** dataset:

1. **Registry Data**: Place your real Excel files in `data/xlsx_files/`
2. **Audio Data**: Place real audio files and LLM results in `data/audio_files/`
3. **Text Data**: Place any raw text files in `data/txt_files/` for preprocessing
4. **Configuration**: Adjust parameters as needed:
   - `TIME_WINDOW_MINUTES` for matching window
   - `MANUAL_SAMPLE_SIZE` for quality control
5. **Run Pipeline**: Execute the complete processing workflow

### Dataset Timeline
- **Phase 1**: 2023-2024 dataset processing
- **Phase 2**: Scale to full historical dataset
- **Phase 3**: Real-time processing capabilities

⚠️ **Important**: Remove or replace the synthetic sample data before processing real data.

## File Structure

```
ems_matching/
├── trauma_data_processor_simple.py  # Main script (one file does it all!)
├── requirements.txt                 # Python dependencies
├── README.md                       # Complete documentation
├── data/                           # Synthetic sample data
│   ├── xlsx_files/                # Synthetic Excel files with registry data
│   ├── audio_files/               # Synthetic audio files and JSON data
│   └── txt_files/                 # TXT files for preprocessing (NEW!)
└── output/                         # Output files (auto-generated)
    ├── trauma_data_unified.xlsx    # Main results
    ├── trauma_data_unified.csv     # CSV version
    └── other output files...
```

⚠️ **Note**: All data in the `data/` folder is **synthetic** and for demonstration purposes only.


