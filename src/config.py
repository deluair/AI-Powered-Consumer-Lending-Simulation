import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Project Root Directory
# Assuming this config.py is in lending_simulator_project/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Path Configurations ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create dirs if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = LOGS_DIR / "simulation.log"

# --- Synthetic Data Generation Parameters ---
NUM_TRADITIONAL_PROFILES = int(os.getenv("NUM_TRADITIONAL_PROFILES", 4000))
NUM_THIN_FILE_PROFILES = int(os.getenv("NUM_THIN_FILE_PROFILES", 3500))
NUM_CREDIT_INVISIBLE_PROFILES = int(os.getenv("NUM_CREDIT_INVISIBLE_PROFILES", 2500))
TOTAL_SYNTHETIC_PROFILES = NUM_TRADITIONAL_PROFILES + NUM_THIN_FILE_PROFILES + NUM_CREDIT_INVISIBLE_PROFILES

# --- Model Configuration ---
DEFAULT_MODEL_FILENAME = "ensemble_model_v1.joblib"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_FILENAME

# --- API Configuration ---
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"  # For uvicorn reload
PROJECT_NAME = os.getenv("PROJECT_NAME", "AI-Powered Consumer Lending Simulation")
VERSION = os.getenv("VERSION", "0.1.0")

# --- CLI & Specific Module Configurations (add as empty dicts for now) ---
SYNTHETIC_DATA_GENERATION_CONFIG = {}
TRADITIONAL_MODEL_CONFIG = {}
ALTERNATIVE_MODEL_CONFIG = {}
ENSEMBLE_MODEL_CONFIG = {}
SIMULATION_CONFIG = {}

# --- Database (Example - to be configured if used) ---
# DB_TYPE = os.getenv("DB_TYPE")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# MONGO_URI = os.getenv("MONGO_URI")

# --- Feature Engineering ---
# Example: List of features to use, or parameters for transformations
# SELECTED_FEATURES = ['feature1', 'feature2', ...]

# --- Economic Stress Scenarios ---
# Define parameters for different economic scenarios
ECONOMIC_SCENARIOS = {
    "baseline": {"unemployment_rate_change": 0, "gdp_growth_change": 0},
    "recession": {"unemployment_rate_change": 0.05, "gdp_growth_change": -0.02},
    "inflation_stress": {"interest_rate_increase": 0.02, "inflation_increase": 0.03}
}

# --- Explainability ---
# Configuration for SHAP, LIME, etc.
# SHAP_BACKGROUND_DATA_SAMPLE_SIZE = 100

# --- Bias and Fairness ---
# Protected attributes for fairness analysis
# PROTECTED_ATTRIBUTES = ['age_group', 'gender', 'race'] # Example

print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Models Directory: {MODELS_DIR}")