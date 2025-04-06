# models/train_delay_model.py
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_classifier(csv_path):
    """Train an XGBoost classifier for route delay prediction."""
    data = pd.read_csv(csv_path)
    X = data[['speed', 'route_variance']]
    y = data['delay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                              use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Delay model accuracy: {acc}")
    
    joblib.dump(model, "models/delay_model.pkl")
    return model

if __name__ == "__main__":
    train_classifier("data/merged_data.csv")
