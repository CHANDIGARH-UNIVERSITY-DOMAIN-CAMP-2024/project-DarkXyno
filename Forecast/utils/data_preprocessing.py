import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    print("Loading data...")
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the data by handling missing values and renaming columns."""
    print("Cleaning data...")
    # Drop rows with missing values
    df = df.dropna()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates additional features for better prediction."""
    print("Engineering features...")
    # Convert 'DATE_TIME' to datetime and extract useful components
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df['Hour'] = df['DATE_TIME'].dt.hour
    df['DayOfYear'] = df['DATE_TIME'].dt.dayofyear
    df['Month'] = df['DATE_TIME'].dt.month
    return df

if __name__ == "__main__":
    # Load the dataset
    file_path = "../data/raw_data.csv"  # Adjust path if needed
    try:
        df = load_data(file_path)
        df = clean_data(df)
        df = engineer_features(df)
        print("Preprocessing completed. Here's a sample:")
        print(df.head())
        
        # Save the processed data to a new file
        output_path = "../data/processed_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")