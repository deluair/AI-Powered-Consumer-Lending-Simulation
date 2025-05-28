import pandas as pd
from ..config import RAW_DATA_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)

def load_alternative_data(file_name: str = "alternative_data.csv") -> pd.DataFrame:
    """
    Loads alternative data (e.g., utility payments, digital footprint) from the raw data directory.
    This is a placeholder and needs to be adapted to the actual data format(s).
    Alternative data might come from multiple files or APIs.
    """
    try:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            logger.warning(f"Alternative data file not found: {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()
        
        # Assuming CSV for now, might need to handle JSON, XML, database connections, etc.
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded alternative data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading alternative data from {file_name}: {e}")
        return pd.DataFrame()

def load_utility_payments(file_name: str = "utility_payments.csv") -> pd.DataFrame:
    logger.info(f"Attempting to load utility payment data from {file_name}")
    return load_alternative_data(file_name) # Reuse generic loader for now

def load_digital_footprint_data(file_name: str = "digital_footprint.csv") -> pd.DataFrame:
    logger.info(f"Attempting to load digital footprint data from {file_name}")
    return load_alternative_data(file_name) # Reuse generic loader for now

if __name__ == '__main__':
    # Example: Create dummy CSVs for testing if they don't exist
    dummy_alt_file_path = RAW_DATA_DIR / "alternative_data.csv"
    if not dummy_alt_file_path.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dummy_data = {
            'user_id': ['TRAD_001', 'THIN_001', 'INVIS_001'],
            'utility_payment_rating': ['Good', 'Excellent', 'Fair'],
            'digital_footprint_score': [7, 8, 5]
        }
        pd.DataFrame(dummy_data).to_csv(dummy_alt_file_path, index=False)
        logger.info(f"Created dummy alternative data at {dummy_alt_file_path}")

    alternative_df = load_alternative_data()
    if not alternative_df.empty:
        print("Alternative Data Sample:")
        print(alternative_df.head())
    else:
        print("Failed to load alternative data or file does not exist.")