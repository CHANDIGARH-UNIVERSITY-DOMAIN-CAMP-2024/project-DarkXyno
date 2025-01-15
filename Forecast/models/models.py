from sklearn.ensemble import RandomForestRegressor

def create_model():
    """Creates a RandomForest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return model
