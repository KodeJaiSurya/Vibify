import tensorflow_data_validation as tfdv
import pandas as pd
import numpy

def getAnomalies(array: numpy.array):

    # Generate data statistics from the input DataFrame, Infer a schema from statistics, and identify anomalies
    df = pd.DataFrame(array)
    data_stats = tfdv.generate_statistics_from_dataframe(df)
    schema = tfdv.infer_schema(statistics=data_stats)
    anomalies = tfdv.validate_statistics(statistics=data_stats, schema=schema)
    
    # Check if any anomalies are detected
    if anomalies.anomaly_info:
        tfdv.display_anomalies(anomalies)
        print("Anomalies detected in the data!")
    else:
        print("No anomalies detected.")