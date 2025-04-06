# models/train_fuel_model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

def train_model(csv_path):
    """Train a LightGBM regressor for predicting fuel efficiency."""
    data = pd.read_csv(csv_path)
    X = data[['avg_fuel', 'speed', 'rpm']]
    y = data['fuel_efficiency']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = lgb.Dataset(X_train, label=y_train)
    valid_ds = lgb.Dataset(X_test, label=y_test)
    
    params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1}
    model = lgb.train(params, train_ds, valid_sets=[valid_ds], num_boost_round=100, early_stopping_rounds=10)
    
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Fuel model MSE: {mse}")
    
    joblib.dump(model, "models/fuel_model.pkl")
    return model

if __name__ == "__main__":
    train_model("data/merged_data.csv")
