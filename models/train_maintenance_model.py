# models/train_maintenance_model.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_maintenance(csv_path):
    """Train a LightGBM binary classifier for maintenance risk prediction."""
    data = pd.read_csv(csv_path)
    X = data[['engine_temp', 'vibration', 'avg_fuel']]
    y = data['maintenance_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = lgb.Dataset(X_train, label=y_train)
    valid_ds = lgb.Dataset(X_test, label=y_test)
    
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1}
    model = lgb.train(params, train_ds, valid_sets=[valid_ds], num_boost_round=100, early_stopping_rounds=10)
    
    preds = model.predict(X_test)
    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    acc = accuracy_score(y_test, preds_bin)
    print(f"Maintenance model accuracy: {acc}")
    
    joblib.dump(model, "models/maintenance_model.pkl")
    return model

if __name__ == "__main__":
    train_maintenance("data/merged_data.csv")
