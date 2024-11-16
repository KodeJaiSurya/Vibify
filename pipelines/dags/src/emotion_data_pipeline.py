from pathlib import Path
from typing import List, Tuple
import logging

from pipelines.dags.src.emotion_data_downloader import DataDownloader
from pipelines.dags.src.emotion_data_processor import DataProcessor
from pipelines.dags.src.emotion_data_aggregator import DataAggregator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)

def download_emotion_data(file_id: str) -> str:
    """Task to download emotion data"""
    logger.info("Starting download task for emotion data")
    downloader = DataDownloader()
    try:
        file_path = downloader.download_from_gdrive(file_id)
        logger.info("Emotion data downloaded successfully.")
        return file_path
    except Exception as e:
        logger.error(f"Error in download_emotion_data: {e}")
        raise

def process_emotion_data(file_path: str, chunk_size: int = 1000) -> List[Tuple[Path, Path]]:
    """Task to process emotion data in chunks"""
    logger.info("Starting processing task for emotion data")
    processor = DataProcessor(chunk_size=chunk_size)
    try:
        chunk_paths = processor.process_all_chunks(file_path)
        logger.info("Emotion data processed into chunks successfully.")
        return chunk_paths
    except Exception as e:
        logger.error(f"Error in process_emotion_data: {e}")
        raise

def aggregate_filtered_data(chunk_paths: List[Tuple[Path, Path]]) -> bool:
    """Task to combine chunks and save final data"""
    logger.info("Starting aggregation task for emotion data")
    aggregator = DataAggregator()
    try:
        X, y = aggregator.combine_chunks(chunk_paths)
        success = aggregator.save_final(X, y)
        aggregator.cleanup_chunks(chunk_paths)
        if success:
            logger.info("Emotion data aggregation and saving completed successfully.")
        else:
            logger.warning("Emotion data aggregation completed but saving failed.")
        return success
    except Exception as e:
        logger.error(f"Error in aggregate_filtered_data: {e}")
        raise