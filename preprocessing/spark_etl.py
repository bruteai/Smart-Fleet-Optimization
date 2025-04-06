# preprocessing/spark_etl.py
from pyspark.sql import SparkSession

def get_spark(app_name="SmartFleetETL"):
    """Initialize a Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_csv(spark, path):
    """Read a CSV file into a Spark DataFrame."""
    return spark.read.csv(path, header=True, inferSchema=True)

def run_etl():
    """Perform a simple ETL operation on CAN and GPS data."""
    spark = get_spark()
    can_df = load_csv(spark, "data/can_bus.csv")
    gps_df = load_csv(spark, "data/gps.csv")
    can_df.createOrReplaceTempView("can")
    gps_df.createOrReplaceTempView("gps")
    query = """
      SELECT c.*, g.latitude, g.longitude, g.timestamp AS gps_timestamp
      FROM can c LEFT JOIN gps g ON c.timestamp = g.timestamp
    """
    return spark.sql(query)

if __name__ == "__main__":
    df = run_etl()
    df.show(5)
