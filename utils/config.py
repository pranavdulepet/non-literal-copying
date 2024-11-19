# src/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_DIR = BASE_DIR / "results"

# Create necessary directories
for dir_path in [DATASETS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)
    
# Dataset subdirectories
ORIGINAL_TEXTS_DIR = DATASETS_DIR / "original_texts"
PREPROCESSED_TEXTS_DIR = DATASETS_DIR / "preprocessed_texts"
PROMPTS_DIR = DATASETS_DIR / "prompts"
GENERATED_TEXTS_DIR = DATASETS_DIR / "generated_texts"

# Results subdirectories
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

# Create all subdirectories
for dir_path in [ORIGINAL_TEXTS_DIR, PREPROCESSED_TEXTS_DIR, 
                 PROMPTS_DIR, GENERATED_TEXTS_DIR,
                 ANALYSIS_RESULTS_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

# Model parameters
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 1000
TOP_P = 0.9

# Experiment settings
NUM_BOOKS = 5
EXCERPT_LENGTH = 500
NUM_TOPICS = 10

# Validation
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
