# data_ingestion/can_bus.py
import pandas as pd

def load_can_data(path):
    """Read and process CAN bus data from a CSV file.
    
    Expected columns include: timestamp, fuel_rate, speed, rpm, etc.
    """
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

if __name__ == "__main__":
    df = load_can_data("data/can_bus.csv")
    print(df.head())
