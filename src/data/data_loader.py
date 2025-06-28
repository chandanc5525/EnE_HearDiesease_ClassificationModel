import pandas as pd
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger()

def load_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise CustomException("Failed to load data", e)
