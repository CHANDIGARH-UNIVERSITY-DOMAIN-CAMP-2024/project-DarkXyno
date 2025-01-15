import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

def load_processed_data(file_path: str) -> pd.DataFrame:
    """Loads the processed data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    print("Loading processed data...")
    return pd.read_csv(file_path)

def train_model(df: pd.DataFrame):
    """Trains a Random Forest Regressor model on the processed data."""
    # Features and target variable (e.g., we are predicting 'AC_POWER')
    X = df[['Hour', 'DayOfYear', 'Month']]  # Features
    y = df['AC_POWER']  # Target variable

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

    return model

def save_model(model, filename: str):
    """Save the trained model to a file."""
    print(f"Saving the model to {filename}...")
    joblib.dump(model, filename)
    print(f"Model saved successfully!")

if __name__ == "__main__":
    # Load the processed data
    file_path = "data/processed_data.csv"  # Adjust path if needed
    try:
        df = load_processed_data(file_path)
        model = train_model(df)
        save_model(model, "random_forest_model.joblib")  # Save the trained model
        print("Model training completed!")
    except Exception as e:
        print(f"Error during model training: {e}")
