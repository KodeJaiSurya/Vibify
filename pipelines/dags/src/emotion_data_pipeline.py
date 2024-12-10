import logging
from .emotion_gcs_handler import GCSHandler
from .emotion_data_processor import DataProcessor
from .emotion_data_aggregator import DataAggregator

def init_gcs_handler() -> dict:
    """Initialize GCS handler and verify connection"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    try:
        logger.info("Initializing GCS handler")
        handler = GCSHandler()
        return {"status": "success"}
    except Exception as e:
        logger.error(f"GCS handler initialization error: {e}")
        raise

def process_emotion_data(**context) -> list:
    """Process emotion data chunks"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting processing task")
        gcs_handler = GCSHandler()
        processor = DataProcessor(gcs_handler)
        chunk_paths = processor.process_all_chunks("data/raw/facial_expression/fer2013.csv")
        logger.info(f"Generated chunk paths: {chunk_paths}")
        return chunk_paths
    except Exception as e:
        logger.error(f"Data processing error: {e}")
        raise

def aggregate_emotion_data(chunk_paths_func, **context) -> bool:
    """Aggregate processed emotion data"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting aggregation task")
        # Get chunk paths using the provided function
        chunk_paths = chunk_paths_func(**context)
        logger.info(f"Retrieved chunk paths: {chunk_paths}")
        
        gcs_handler = GCSHandler()
        aggregator = DataAggregator(gcs_handler)
        X, y = aggregator.combine_chunks(chunk_paths)
        success = aggregator.save_final(X, y)
        
        if success:
            aggregator.cleanup_chunks(chunk_paths)
            logger.info("Aggregation completed successfully")
        return success
    except Exception as e:
        logger.error(f"Data aggregation error: {e}")
        raise