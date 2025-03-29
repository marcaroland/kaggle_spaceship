import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)
