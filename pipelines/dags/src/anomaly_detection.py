import tensorflow_data_validation as tfdv
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_anomalies(df: pd.DataFrame):
    """
    Detect anomalies in the given DataFrame using TensorFlow Data Validation.
    
    Args:
        df (pd.DataFrame): Input DataFrame to analyze
        
    Raises:
        ValueError: If anomalies are detected in the dataset
    """
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

def process_files(data_dir: str, specific_file: str = ''):
    """
    Process files in the given directory for anomaly detection.
    
    Args:
        data_dir (str): Directory containing the data files
        specific_file (str, optional): Specific file to process. If empty, processes all CSV files.
    """
    if specific_file:
        file_path = os.path.join(data_dir, specific_file)
        if os.path.exists(file_path):
            logger.info(f"Processing specific file: {file_path}")
            df = pd.read_csv(file_path)
            get_anomalies(df)
    else:
        # Process all CSV files in the directory
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(data_dir, file)
                logger.info(f"Processing file: {file_path}")
                df = pd.read_csv(file_path)
                get_anomalies(df)

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./Vibify/pipelines/dags/data"
    specific_file = sys.argv[2] if len(sys.argv) > 2 else ''
    process_files(data_dir, specific_file)

