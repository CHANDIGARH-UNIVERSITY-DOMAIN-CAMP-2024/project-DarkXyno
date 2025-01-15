import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.data_preprocessing import load_data, clean_data, engineer_features, split_data
from models.model import create_model

def train_model(data_path: str, target: str, model_save_path: str):
    """Trains and saves the model."""
    df = load_data(data_path)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = split_data(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model trained with MSE: {mse}")
    
    joblib.dump(model, model_save_path)
    print(f"Model saved at {model_save_path}")
