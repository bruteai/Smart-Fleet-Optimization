# data_ingestion/gps.py
import pandas as pd

def load_gps_data(path):
    """Load GPS trajectory data from a CSV file.
    
    Columns expected: timestamp, latitude, longitude, etc.
    """
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

if __name__ == "__main__":
    df = load_gps_data("data/gps.csv")
    print(df.head())
