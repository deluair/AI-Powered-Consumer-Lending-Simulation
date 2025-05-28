import pandas as pd
from ..config import RAW_DATA_DIR
from ..utils.logger import get_logger

logger = get_logger(__name__)

def load_traditional_data(file_name: str = "traditional_credit_data.csv") -> pd.DataFrame:
    """
    Loads traditional credit data from the raw data directory.
    This is a placeholder and needs to be adapted to the actual data format.
    """
    try:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            logger.error(f"Traditional data file not found: {file_path}")
            # In a real scenario, you might raise an error or return an empty DataFrame
            # For simulation, we can try to load the synthetic one if this is missing
            # or generate a warning and continue.
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded traditional data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading traditional data from {file_name}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example: Create a dummy CSV for testing if it doesn't exist
    dummy_file_path = RAW_DATA_DIR / "traditional_credit_data.csv"
    if not dummy_file_path.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dummy_data = {
            'user_id': ['TRAD_001', 'TRAD_002'],
            'credit_score': [750, 680],
            'loan_amount': [10000, 15000],
            'defaulted': [0, 1]
        }
        pd.DataFrame(dummy_data).to_csv(dummy_file_path, index=False)
        logger.info(f"Created dummy traditional data at {dummy_file_path}")

    traditional_df = load_traditional_data()
    if not traditional_df.empty:
        print("Traditional Data Sample:")
        print(traditional_df.head())
    else:
        print("Failed to load traditional data or file does not exist.")