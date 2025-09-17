#!/usr/bin/env python3
"""
Simple Trauma Data Processing Script for Beginners

This is a simplified all-in-one script that automatically processes trauma activation data.
Just run it and it will:
1. Check dependencies
2. Create sample data if needed
3. Process the data
4. Show results

Usage:
    python trauma_data_processor_simple.py

Requirements:
    pip install pandas numpy openpyxl xlrd
"""

import pandas as pd
import numpy as np
import os
import re
import json
import wave
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any
import importlib

# =============================================================================
# CONFIGURATION
# =============================================================================

# File paths
XLSX_FOLDER = "data/xlsx_files"
AUDIO_FOLDER = "data/audio_files"
TXT_FOLDER = "data/txt_files"  # New folder for .txt files
OUTPUT_FOLDER = "output"

# Processing parameters
TIME_WINDOW_MINUTES = 20
MANUAL_SAMPLE_SIZE = 10

# Matching weights
MATCHING_WEIGHTS = {
    'time_proximity': 0.3,
    'sex_match': 0.3,
    'age_match': 0.2,
    'mechanism_match': 0.1,
    'injury_match': 0.1
}

# Age matching thresholds
AGE_MATCH_THRESHOLDS = {'exact': 5, 'partial': 10}

# File extensions
AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.aac', '.flac']
TXT_EXTENSIONS = ['.txt']

# Timestamp patterns
TIMESTAMP_PATTERNS = [
    r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',
    r'(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})',
    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
    r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})',
    r'(\d{8}_\d{6})',
]

TIMESTAMP_FORMATS = [
    '%Y-%m-%d_%H-%M-%S',
    '%Y%m%d_%H%M%S',
    '%Y-%m-%d %H:%M:%S',
    '%m-%d-%Y_%H-%M-%S',
    '%Y%m%d_%H%M%S'
]

# =============================================================================
# MAIN PROCESSING CLASS
# =============================================================================

class TraumaDataProcessor:
    """Simple trauma data processor."""
    
    def __init__(self):
        """Initialize the processor."""
        self.xlsx_folder = Path(XLSX_FOLDER)
        self.audio_folder = Path(AUDIO_FOLDER)
        self.txt_folder = Path(TXT_FOLDER)
        self.output_folder = Path(OUTPUT_FOLDER)
        self.output_folder.mkdir(exist_ok=True)
        
        self.time_window = TIME_WINDOW_MINUTES
        self.registry_data = None
        self.audio_data = None
        self.unified_data = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_registry_data(self):
        """Load and filter trauma activation data."""
        self.logger.info("Loading registry data...")
        
        excel_files = list(self.xlsx_folder.glob("*.xlsx")) + list(self.xlsx_folder.glob("*.xls"))
        
        if not excel_files:
            raise FileNotFoundError(f"No Excel files found in {self.xlsx_folder}")
        
        all_data = []
        for file_path in excel_files:
            self.logger.info(f"Processing {file_path.name}")
            df = pd.read_excel(file_path)
            all_data.append(df)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Loaded {len(combined_data)} total records")
        
        # Filter for trauma activations (ED_TTA_TYPE01 > 3)
        df = combined_data.copy()
        df['ED_TTA_TYPE01_clean'] = df['ED_TTA_TYPE01'].astype(str).str.strip()
        df['ED_TTA_TYPE01_clean'] = df['ED_TTA_TYPE01_clean'].replace(['', ' '], np.nan)
        df['ED_TTA_TYPE01_numeric'] = pd.to_numeric(df['ED_TTA_TYPE01_clean'], errors='coerce')
        
        trauma_mask = df['ED_TTA_TYPE01_numeric'] > 3
        self.registry_data = df[trauma_mask].drop(['ED_TTA_TYPE01_clean', 'ED_TTA_TYPE01_numeric'], axis=1)
        
        self.logger.info(f"Filtered to {len(self.registry_data)} trauma activations")
        return self.registry_data
    
    def load_audio_data(self):
        """Load audio file data."""
        self.logger.info("Loading audio file data...")
        
        audio_data = []
        for audio_file in self.audio_folder.rglob("*"):
            if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
                timestamp = self._extract_timestamp_from_filename(audio_file.name)
                if timestamp is None:
                    continue
                
                json_file = audio_file.with_suffix('.json')
                audio_record = {
                    'audio_filename': audio_file.name,
                    'audio_path': str(audio_file),
                    'datetime': timestamp,
                    'json_path': str(json_file) if json_file.exists() else None
                }
                
                if json_file.exists():
                    extracted_data = self._load_extracted_data(json_file)
                    audio_record.update(extracted_data)
                
                audio_data.append(audio_record)
        
        self.audio_data = pd.DataFrame(audio_data)
        self.logger.info(f"Loaded {len(self.audio_data)} audio files")
        return self.audio_data
    
    def _extract_timestamp_from_filename(self, filename: str):
        """Extract datetime from filename."""
        for pattern in TIMESTAMP_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                timestamp_str = match.group(1)
                for fmt in TIMESTAMP_FORMATS:
                    try:
                        return datetime.strptime(timestamp_str, fmt)
                    except ValueError:
                        continue
        return None
    
    def _load_extracted_data(self, json_file: Path):
        """Load data from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {
                'transcript': data.get('transcript', ''),
                'age': data.get('age', None),
                'sex': data.get('sex', ''),
                'mechanism': data.get('mechanism', ''),
                'injuries': data.get('injuries', ''),
                'activation_page': data.get('activation_page', ''),
            }
        except Exception as e:
            self.logger.error(f"Error loading JSON file {json_file}: {e}")
            return {'transcript': '', 'age': None, 'sex': '', 'mechanism': '', 'injuries': '', 'activation_page': ''}
    
    def match_data(self):
        """Match registry data with audio data."""
        self.logger.info("Matching registry data with audio data...")
        
        matched_data = []
        for _, registry_row in self.registry_data.iterrows():
            arrival_time = self._parse_arrival_time(registry_row)
            if arrival_time is None:
                continue
            
            audio_matches = self._find_audio_matches(arrival_time, registry_row)
            
            record = registry_row.to_dict()
            record.update({
                'arrival_time': arrival_time,
                'audio_matches': audio_matches,
                'best_match': audio_matches[0] if audio_matches else None,
                'num_matches': len(audio_matches)
            })
            matched_data.append(record)
        
        self.unified_data = pd.DataFrame(matched_data)
        self.logger.info(f"Created unified dataset with {len(self.unified_data)} records")
        return self.unified_data
    
    def _parse_arrival_time(self, registry_row):
        """Parse arrival time from registry data."""
        arrival_column = 'Patient Arrival Date & Time'
        if arrival_column not in registry_row:
            for col in ['Arrival Date & Time', 'Arrival_Date_Time', 'arrival_time']:
                if col in registry_row:
                    arrival_column = col
                    break
            else:
                return None
        
        arrival_time = registry_row[arrival_column]
        if pd.isna(arrival_time):
            return None
        
        if isinstance(arrival_time, datetime):
            return arrival_time
        
        try:
            return pd.to_datetime(arrival_time)
        except:
            return None
    
    def _find_audio_matches(self, arrival_time, registry_row):
        """Find matching audio files."""
        audio_matches = []
        
        for _, audio_row in self.audio_data.iterrows():
            audio_time = audio_row['datetime']
            time_diff = abs((audio_time - arrival_time).total_seconds() / 60)
            
            if time_diff <= self.time_window:
                match_score = self._calculate_match_score(registry_row, audio_row, time_diff)
                audio_matches.append({
                    'audio_filename': audio_row['audio_filename'],
                    'audio_path': audio_row['audio_path'],
                    'audio_datetime': audio_time,
                    'time_diff_minutes': time_diff,
                    'match_score': match_score,
                    'extracted_data': {
                        'age': audio_row.get('age'),
                        'sex': audio_row.get('sex', ''),
                        'mechanism': audio_row.get('mechanism', ''),
                        'injuries': audio_row.get('injuries', ''),
                        'transcript': audio_row.get('transcript', ''),
                        'activation_page': audio_row.get('activation_page', '')
                    }
                })
        
        audio_matches.sort(key=lambda x: x['match_score'], reverse=True)
        return audio_matches
    
    def _calculate_match_score(self, registry_row, audio_row, time_diff):
        """Calculate match score."""
        score = 0.0
        
        # Time proximity
        time_score = max(0, 1 - (time_diff / self.time_window))
        score += time_score * MATCHING_WEIGHTS['time_proximity']
        
        # Sex match
        registry_sex = str(registry_row.get('Sex', '')).strip().lower()
        audio_sex = str(audio_row.get('sex', '')).strip().lower()
        if registry_sex and audio_sex and registry_sex == audio_sex:
            score += MATCHING_WEIGHTS['sex_match']
        
        # Age match
        registry_age = registry_row.get('Age', None)
        audio_age = audio_row.get('age', None)
        if registry_age is not None and audio_age is not None:
            try:
                age_diff = abs(float(registry_age) - float(audio_age))
                if age_diff <= AGE_MATCH_THRESHOLDS['exact']:
                    score += MATCHING_WEIGHTS['age_match']
                elif age_diff <= AGE_MATCH_THRESHOLDS['partial']:
                    score += MATCHING_WEIGHTS['age_match'] * 0.5
            except:
                pass
        
        # Mechanism and injury matching
        mechanism_score = self._calculate_keyword_match_score(
            registry_row.get('Mechanism', ''), audio_row.get('mechanism', ''))
        score += mechanism_score * MATCHING_WEIGHTS['mechanism_match']
        
        injury_score = self._calculate_keyword_match_score(
            registry_row.get('Injuries', ''), audio_row.get('injuries', ''))
        score += injury_score * MATCHING_WEIGHTS['injury_match']
        
        return score
    
    def _calculate_keyword_match_score(self, text1: str, text2: str):
        """Calculate keyword match score."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def save_results(self):
        """Save all results."""
        if self.registry_data is not None:
            registry_file = self.output_folder / "trauma_data_registry_filtered.xlsx"
            self.registry_data.to_excel(registry_file, index=False)
            self.logger.info(f"Saved registry data to {registry_file}")
        
        if self.audio_data is not None:
            audio_file = self.output_folder / "trauma_data_audio_data.xlsx"
            self.audio_data.to_excel(audio_file, index=False)
            self.logger.info(f"Saved audio data to {audio_file}")
        
        if self.unified_data is not None:
            unified_file = self.output_folder / "trauma_data_unified.xlsx"
            self.unified_data.to_excel(unified_file, index=False)
            self.logger.info(f"Saved unified data to {unified_file}")
            
            unified_csv = self.output_folder / "trauma_data_unified.csv"
            self.unified_data.to_csv(unified_csv, index=False)
            self.logger.info(f"Saved unified data to {unified_csv}")
    
    def run_full_pipeline(self):
        """Run the complete processing pipeline."""
        self.logger.info("Starting trauma data processing...")
        
        # Load data
        self.load_registry_data()
        self.load_audio_data()
        
        # Match data
        self.match_data()
        
        # Save results
        self.save_results()
        
        # Print summary
        self._print_summary()
        
        self.logger.info("Processing completed successfully!")
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        
        if self.registry_data is not None:
            print(f"Registry records loaded: {len(self.registry_data)}")
        
        if self.audio_data is not None:
            print(f"Audio files processed: {len(self.audio_data)}")
        
        if self.unified_data is not None:
            print(f"Unified records created: {len(self.unified_data)}")
            
            with_matches = len(self.unified_data[self.unified_data['num_matches'] > 0])
            without_matches = len(self.unified_data[self.unified_data['num_matches'] == 0])
            
            print(f"Records with audio matches: {with_matches}")
            print(f"Records without audio matches: {without_matches}")
            
            if with_matches > 0:
                avg_matches = self.unified_data[self.unified_data['num_matches'] > 0]['num_matches'].mean()
                print(f"Average matches per record: {avg_matches:.2f}")
        
        print("="*50)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['pandas', 'numpy', 'openpyxl', 'xlrd']
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install pandas numpy openpyxl xlrd")
        return False
    
    return True

def create_tone_wav(filename, frequency=440, duration=5, sample_rate=44100):
    """Create a simple tone WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = np.sin(2 * np.pi * frequency * t)
    wave_data = (wave_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())

def preprocess_txt_files():
    """Preprocess .txt files and convert them to JSON format."""
    print("Preprocessing .txt files...")
    
    txt_folder = Path(TXT_FOLDER)
    if not txt_folder.exists():
        print(f"TXT folder {TXT_FOLDER} does not exist. Creating it...")
        txt_folder.mkdir(parents=True, exist_ok=True)
        return
    
    # Find all .txt files
    txt_files = []
    for file_path in txt_folder.rglob("*"):
        if file_path.suffix.lower() in TXT_EXTENSIONS:
            txt_files.append(file_path)
    
    if not txt_files:
        print(f"No .txt files found in {TXT_FOLDER}")
        return
    
    print(f"Found {len(txt_files)} .txt files to process")
    
    # Process each .txt file
    for txt_file in txt_files:
        try:
            # Extract timestamp from filename
            timestamp = extract_timestamp_from_filename(txt_file.name)
            if timestamp is None:
                print(f"Warning: Could not extract timestamp from {txt_file.name}, skipping...")
                continue
            
            # Read the .txt file content
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Parse the content to extract structured data
            parsed_data = parse_txt_content(content)
            
            # Create corresponding JSON file in audio_files folder
            json_filename = txt_file.stem + '.json'
            json_file = Path(AUDIO_FOLDER) / json_filename
            
            # Ensure audio_files directory exists
            Path(AUDIO_FOLDER).mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Processed {txt_file.name} -> {json_filename}")
            
        except Exception as e:
            print(f"✗ Error processing {txt_file.name}: {e}")
    
    print("TXT preprocessing completed!")

def extract_timestamp_from_filename(filename: str):
    """Extract datetime from filename (same logic as in the class)."""
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, filename)
        if match:
            timestamp_str = match.group(1)
            for fmt in TIMESTAMP_FORMATS:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
    return None

def parse_txt_content(content: str):
    """Parse .txt content and extract structured data."""
    # Initialize default values
    parsed_data = {
        'transcript': content,
        'age': None,
        'sex': '',
        'mechanism': '',
        'injuries': '',
        'activation_page': ''
    }
    
    # Try to extract structured information using regex patterns
    content_lower = content.lower()
    
    # Extract age (look for patterns like "25-year-old", "age 25", "25 yo", etc.)
    age_patterns = [
        r'(\d+)[-\s]year[-\s]old',
        r'age[:\s]+(\d+)',
        r'(\d+)\s*yo\b',
        r'(\d+)\s*years?\s*old'
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, content_lower)
        if match:
            try:
                parsed_data['age'] = int(match.group(1))
                break
            except ValueError:
                continue
    
    # Extract sex (look for M/F, Male/Female, etc.)
    sex_patterns = [
        r'\b(male|m)\b',
        r'\b(female|f)\b'
    ]
    
    for pattern in sex_patterns:
        match = re.search(pattern, content_lower)
        if match:
            sex_text = match.group(1).lower()
            if sex_text in ['male', 'm']:
                parsed_data['sex'] = 'M'
            elif sex_text in ['female', 'f']:
                parsed_data['sex'] = 'F'
            break
    
    # Extract mechanism (common trauma mechanisms)
    mechanism_keywords = {
        'motor vehicle accident': ['mva', 'motor vehicle', 'car accident', 'vehicle accident', 'crash'],
        'fall from height': ['fall', 'height', 'high fall', 'fell from'],
        'penetrating injury': ['gunshot', 'stab', 'penetrating', 'gsw', 'knife'],
        'blunt trauma': ['blunt', 'hit', 'struck', 'beaten'],
        'assault': ['assault', 'attack', 'fight', 'beaten'],
        'industrial accident': ['industrial', 'workplace', 'machinery', 'equipment']
    }
    
    for mechanism, keywords in mechanism_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            parsed_data['mechanism'] = mechanism
            break
    
    # Extract injuries (common injury patterns)
    injury_keywords = [
        'head injury', 'chest trauma', 'abdominal trauma', 'spinal injury', 'fracture',
        'internal bleeding', 'burns', 'laceration', 'contusion', 'concussion',
        'rib fracture', 'pelvic fracture', 'femur fracture', 'skull fracture'
    ]
    
    found_injuries = []
    for injury in injury_keywords:
        if injury in content_lower:
            found_injuries.append(injury.title())
    
    if found_injuries:
        parsed_data['injuries'] = ', '.join(found_injuries)
    
    # Generate activation page based on content
    if parsed_data['age'] and parsed_data['sex'] and parsed_data['mechanism']:
        activation_text = f"Trauma activation for {parsed_data['age']}-year-old {parsed_data['sex']} with {parsed_data['mechanism']}"
        if parsed_data['injuries']:
            activation_text += f". Injuries: {parsed_data['injuries']}"
        activation_text += ". Trauma team activation required."
        parsed_data['activation_page'] = activation_text
    
    return parsed_data

def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    
    # Create directories
    Path(XLSX_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(AUDIO_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Create sample Excel data
    sample_data = {
        'Patient ID': [f'P{i:03d}' for i in range(1, 21)],
        'ED_TTA_TYPE01': [4, 5, 2, 6, 4, 7, 3, 5, 4, 6, 5, 4, 7, 6, 5, 4, 6, 5, 4, 7],
        'Patient Arrival Date & Time': [
            '2023-01-15 14:25:30', '2023-01-15 15:10:45', '2023-01-15 16:30:15',
            '2023-01-15 17:45:20', '2023-01-15 18:20:10', '2023-01-15 19:15:35',
            '2023-01-15 20:00:50', '2023-01-15 21:30:25', '2023-01-15 22:45:15',
            '2023-01-15 23:20:40', '2023-01-16 00:15:30', '2023-01-16 01:30:45',
            '2023-01-16 02:45:20', '2023-01-16 03:20:15', '2023-01-16 04:35:30',
            '2023-01-16 05:50:25', '2023-01-16 06:15:40', '2023-01-16 07:30:15',
            '2023-01-16 08:45:50', '2023-01-16 09:20:35'
        ],
        'Sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'Age': [25, 32, 45, 28, 35, 22, 50, 29, 41, 33, 27, 38, 44, 31, 26, 29, 37, 24, 42, 35],
        'Mechanism': [
            'Motor vehicle accident', 'Fall from height', 'Penetrating injury', 'Blunt trauma',
            'Motor vehicle accident', 'Assault', 'Industrial accident', 'Motor vehicle accident',
            'Penetrating injury', 'Fall from height', 'Motor vehicle accident', 'Blunt trauma',
            'Assault', 'Industrial accident', 'Motor vehicle accident', 'Penetrating injury',
            'Fall from height', 'Blunt trauma', 'Motor vehicle accident', 'Assault'
        ],
        'Injuries': [
            'Head injury and chest trauma', 'Multiple fractures and internal bleeding',
            'Gunshot wound to abdomen', 'Head injury and spinal cord injury',
            'Multiple trauma and burns', 'Head injury and facial fractures',
            'Crush injury to lower extremities', 'Head injury and multiple fractures',
            'Stab wound to chest', 'Spinal injury and head trauma',
            'Head injury and internal bleeding', 'Multiple rib fractures and lung injury',
            'Head injury and facial trauma', 'Crush injury and burns',
            'Head injury and spinal cord injury', 'Gunshot wound to head',
            'Multiple fractures and head injury', 'Head injury and abdominal trauma',
            'Multiple trauma and head injury', 'Head injury and neck trauma'
        ],
        'Chief Complaint': [
            'Unconscious after MVA', 'High fall injury', 'GSW to abdomen',
            'Blunt head trauma', 'MVA with burns', 'Assault victim',
            'Industrial crush injury', 'MVA head injury', 'Stab wound chest',
            'High fall with spinal injury', 'MVA unconscious', 'Blunt chest trauma',
            'Assault head injury', 'Industrial accident', 'MVA spinal injury',
            'GSW to head', 'High fall multiple injuries', 'Blunt head and abdomen',
            'MVA multiple trauma', 'Assault neck injury'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    excel_file = Path(XLSX_FOLDER) / "trauma_registry_sample.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"Created sample Excel file: {excel_file}")
    
    # Create sample JSON files
    json_templates = [
        {
            "transcript": "This is a 25-year-old male patient involved in a motor vehicle accident. Patient was unconscious at the scene. Head injury with possible skull fracture. Chest trauma with rib fractures. Patient is currently intubated and on mechanical ventilation. Blood pressure is stable. Heart rate is elevated at 110. Patient needs immediate trauma activation.",
            "age": 25, "sex": "M", "mechanism": "Motor vehicle accident", "injuries": "Head injury, chest trauma, rib fractures",
            "activation_page": "STAT trauma activation for 25-year-old male MVA victim with head injury and chest trauma. Patient unconscious, intubated, stable vitals. Immediate trauma team activation required."
        },
        {
            "transcript": "32-year-old female patient fell from height, approximately 20 feet. Multiple fractures including femur, pelvis, and spine. Internal bleeding suspected. Patient is conscious but in severe pain. Blood pressure is dropping. Heart rate is 130. Need immediate trauma activation for this high fall victim.",
            "age": 32, "sex": "F", "mechanism": "Fall from height", "injuries": "Multiple fractures, internal bleeding, spinal injury",
            "activation_page": "STAT trauma activation for 32-year-old female high fall victim. Multiple fractures including femur, pelvis, spine. Internal bleeding suspected. Blood pressure dropping. Immediate trauma team activation required."
        }
    ]
    
    timestamps = [
        "2023-01-15_14-30-25", "2023-01-15_15-15-30", "2023-01-15_16-35-20", "2023-01-15_17-50-15",
        "2023-01-15_18-25-05", "2023-01-15_19-20-40", "2023-01-15_20-05-55", "2023-01-15_21-35-30",
        "2023-01-15_22-50-20", "2023-01-15_23-25-45", "2023-01-16_00-20-35", "2023-01-16_01-35-50",
        "2023-01-16_02-50-25", "2023-01-16_03-25-20", "2023-01-16_04-40-35", "2023-01-16_05-55-30",
        "2023-01-16_06-20-45", "2023-01-16_07-35-20", "2023-01-16_08-50-55", "2023-01-16_09-25-40"
    ]
    
    for i, timestamp in enumerate(timestamps):
        json_file = Path(AUDIO_FOLDER) / f"{timestamp}.json"
        template = json_templates[i % len(json_templates)]
        with open(json_file, 'w') as f:
            json.dump(template, f, indent=2)
    
    print(f"Created {len(timestamps)} sample JSON files")
    
    # Create sample audio files
    frequencies = [220, 330, 440, 550, 660, 770, 880, 990, 1100, 1320]
    for i, timestamp in enumerate(timestamps):
        wav_file = Path(AUDIO_FOLDER) / f"{timestamp}.wav"
        freq = frequencies[i % len(frequencies)]
        duration = 3 + (i % 6)  # 3-8 seconds
        create_tone_wav(str(wav_file), frequency=freq, duration=duration)
    
    print(f"Created {len(timestamps)} sample audio files")

def view_results():
    """View processing results."""
    unified_file = Path(f"{OUTPUT_FOLDER}/trauma_data_unified.csv")
    
    if not unified_file.exists():
        print("No results found. Run processing first.")
        return
    
    df = pd.read_csv(unified_file)
    
    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)
    
    print(f"Total records processed: {len(df)}")
    print(f"Records with matches: {len(df[df['num_matches'] > 0])}")
    print(f"Records without matches: {len(df[df['num_matches'] == 0])}")
    
    if len(df[df['num_matches'] > 0]) > 0:
        avg_matches = df[df['num_matches'] > 0]['num_matches'].mean()
        print(f"Average matches per record: {avg_matches:.2f}")
    
    print("\nSample matches:")
    for i, row in df.head(3).iterrows():
        print(f"\nRecord {i+1}: {row['Patient ID']}")
        print(f"  Registry: {row['Sex']}, {row['Age']}yo, {row['Mechanism']}")
        print(f"  Arrival: {row['Patient Arrival Date & Time']}")
        print(f"  Matches: {row['num_matches']}")
        
        if row['num_matches'] > 0:
            print(f"  ✓ Found {row['num_matches']} audio match(es)")
            # Note: Detailed match info is available in the Excel/CSV files

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function - runs everything automatically."""
    print("=" * 60)
    print("TRAUMA DATA PROCESSOR - SIMPLE VERSION")
    print("=" * 60)
    print("This script will automatically process trauma data.")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        print("Please install missing packages and try again.")
        return
    print("✓ All dependencies are installed!")
    
    # Preprocess .txt files if they exist
    print("\n2. Checking for .txt files to preprocess...")
    if Path(TXT_FOLDER).exists() and list(Path(TXT_FOLDER).glob("*.txt")):
        print("Found .txt files, preprocessing...")
        preprocess_txt_files()
        print("✓ TXT preprocessing completed!")
    else:
        print("✓ No .txt files found, skipping preprocessing...")
    
    # Create sample data if needed
    # print("\n3. Checking for sample data...")
    # if not Path(XLSX_FOLDER).exists() or not list(Path(XLSX_FOLDER).glob("*.xlsx")):
    #     print("Creating sample data...")
    #     create_sample_data()
    #     print("✓ Sample data created!")
    # else:
    #     print("✓ Sample data already exists!")
    
    # Process data
    print("\n3. Processing trauma data...")
    try:
        processor = TraumaDataProcessor()
        processor.run_full_pipeline()
        print("✓ Processing completed successfully!")
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        return
    
    # Show results
    print("Showing results...")
    view_results()
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("="*60)
    print("1. output/trauma_data_registry_filtered.xlsx - Filtered registry data")
    print("2. output/trauma_data_audio_data.xlsx - Processed audio data")
    print("3. output/trauma_data_unified.xlsx - Unified data with matches")
    print("4. output/trauma_data_unified.csv - CSV version of unified data")
    print("\n✓ All done! Check the output folder for results.")

if __name__ == "__main__":
    main()
