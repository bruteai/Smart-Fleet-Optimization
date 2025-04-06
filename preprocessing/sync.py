# preprocessing/sync.py
import pandas as pd

def sync_data(can_df, gps_df):
    """Synchronize CAN bus and GPS data based on timestamps using an as-of merge."""
    can_df = can_df.sort_values('timestamp')
    gps_df = gps_df.sort_values('timestamp')
    merged = pd.merge_asof(can_df, gps_df, on='timestamp', direction='nearest')
    return merged

if __name__ == "__main__":
    from data_ingestion import can_bus, gps
    can_data = can_bus.load_can_data("data/can_bus.csv")
    gps_data = gps.load_gps_data("data/gps.csv")
    combined = sync_data(can_data, gps_data)
    print(combined.head())
