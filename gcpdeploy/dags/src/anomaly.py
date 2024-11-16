import tensorflow_data_validation as tfdv
import pandas as pd
import logging

# unused variable just to trigger
unused_variableee = "This variable is not used"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def getAnomalies(df: pd.DataFrame):
    logger.info("Starting anomaly detection process.")
    
    # Generate data statistics from the input DataFrame
    logger.info("Generating data statistics.")
    data_stats = tfdv.generate_statistics_from_dataframe(df)
    
    # Infer a schema from the generated statistics
    logger.info("Inferring schema from data statistics.")
    schema = tfdv.infer_schema(statistics=data_stats)
    
    # Validate statistics against the inferred schema
    logger.info("Validating data statistics against schema to find anomalies.")
    anomalies = tfdv.validate_statistics(statistics=data_stats, schema=schema)
    
    # Check if any anomalies are detected and raise error if detected
    if anomalies.anomaly_info:
        logger.warning("Anomalies detected in the data.")
        tfdv.display_anomalies(anomalies)
        raise ValueError("Anomalies detected in the dataset. Workflow aborted.")
    else:
        logger.info("No anomalies detected.")


if __name__ == "__main__":
    getAnomalies(pd.read_csv("dags/data/raw/dataset_fer2013.csv"))
    getAnomalies(pd.read_csv("dags/data/raw/song_dataset.csv"))