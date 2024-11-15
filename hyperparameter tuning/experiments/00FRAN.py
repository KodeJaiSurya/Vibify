# emotion_data_pipeline.py
from emotion_data_pipeline import download_emotion_data, process_emotion_data, aggregate_filtered_data

def main():
    file_id = "1-mzGbBpQxlgSowPHetsofCFjdIoxGNNc"  # File ID for download
    chunk_size = 1000  # Define chunk size for processing

    # Step 1: Download the data
    file_path = None
    try:
        file_path = download_emotion_data(file_id)
        print(f"Downloaded file located at: {file_path}")
    except Exception as e:
        print(f"Download failed: {e}")

    # Step 2: Process the data only if download was successful
    chunk_paths = None
    if file_path:
        try:
            chunk_paths = process_emotion_data(file_path, chunk_size)
            print(f"Data processed into {len(chunk_paths)} chunks at: {chunk_paths}")
        except Exception as e:
            print(f"Processing failed: {e}")

    # Step 3: Aggregate the data only if processing was successful
    if chunk_paths:
        try:
            success = aggregate_filtered_data(chunk_paths)
            if success:
                print("Data aggregation completed and saved successfully.")
            else:
                print("Data aggregation failed during saving.")
        except Exception as e:
            print(f"Aggregation failed: {e}")

if __name__ == "__main__":
    main()
