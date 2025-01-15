import joblib
import pandas as pd
from utils.data_preprocessing import engineer_features

def predict(input_data: pd.DataFrame, model_path: str):
    """Makes predictions using the trained model."""
    model = joblib.load(model_path)
    input_data = engineer_features(input_data)
    predictions = model.predict(input_data)
    return predictions
