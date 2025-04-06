# utils/config.py
import os

DATA_DIR = os.path.join(os.getcwd(), "data")
MODEL_DIR = os.path.join(os.getcwd(), "models")
LOG_DIR = os.path.join(os.getcwd(), "logs")

# Spark settings
SPARK_APP_NAME = "SmartFleetETL"
SPARK_MASTER = "local[*]"

# Model hyperparameters
HYPERPARAMS = {
    "fuel_model": {"num_boost_round": 100, "early_stopping_rounds": 10},
    "delay_model": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
    "maintenance_model": {"num_boost_round": 100, "early_stopping_rounds": 10},
}
