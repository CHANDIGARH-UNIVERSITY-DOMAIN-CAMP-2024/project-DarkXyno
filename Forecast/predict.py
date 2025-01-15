import joblib
import pandas as pd

def make_prediction(input_data):
    """Load the trained model and make a prediction based on the input data."""
    model = joblib.load("random_forest_model.joblib")

    # Convert input data to a DataFrame with correct feature names
    input_df = pd.DataFrame([input_data], columns=['Hour', 'DayOfYear', 'Month'])

    # Make the prediction
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    # Example input data: Hour, DayOfYear, and Month (for prediction)
    input_data = [12, 200, 6]  # Adjust these values as per your data format
    prediction = make_prediction(input_data)
    print(f"Predicted AC_POWER for the input {input_data}: {prediction}")
