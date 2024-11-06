import tensorflow_data_validation as tfdv
import pandas as pd
import numpy as np
import logging
# Added loggers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def getAnomalies(array: np.array):
    logger.info("Starting anomaly detection process.")
    
    # Generate data statistics from the input DataFrame
    logger.info("Generating data statistics.")
    df = pd.DataFrame(array)
    data_stats = tfdv.generate_statistics_from_dataframe(df)
    
    # Infer a schema from the generated statistics
    logger.info("Inferring schema from data statistics.")
    schema = tfdv.infer_schema(statistics=data_stats)
    
    # Validate statistics against the inferred schema
    logger.info("Validating data statistics against schema to find anomalies.")
    anomalies = tfdv.validate_statistics(statistics=data_stats, schema=schema)
    
    # Check if any anomalies are detected
    if anomalies.anomaly_info:
        logger.warning("Anomalies detected in the data.")
        tfdv.display_anomalies(anomalies)
    else:
        logger.info("No anomalies detected.")