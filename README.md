# Vibify: Intelligent Emotion-Based Music Recommendation Using Real-Time Facial Sentiment Analysis

Music has always been a powerful way to express and manage our emotions. With millions of songs at our fingertips, the real challenge isn't finding music but it's finding the right music that matches how we're feeling. We do have music apps that suggest songs based on what we usually listen to, but they often miss the mark when it comes to our ever change in moods.

Why does this matter? Well, music doesn't just entertain us but it can seriously impact our mental health and emotional experiences. It can boost our mood, help us relieve stress, and even make us think better. But when the music we're listening to doesn't match our current emotional vibe, it can feel off and unsatisfying.

This is where Vibify comes in with a cool new approach. Instead of asking you to manually select your mood or guessing based on your listening history, Vibify uses facial recognition to figure out how you're feeling in real-time. It then suggests music that fits your current emotional state. This means you get music recommendations that are more relevant and in tune with your emotions as they change throughout the day, potentially making your listening experience more engaging and emotionally fulfilling.

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Team Members](#team-members)
- [Installation](#installation)
- [Prerequisities](#prerequisities)
- [User Installation](#user-installation)
- [Tools Used](#tools-used-for-mlops)
- [Data Preprocessing](#data-preprocessing)
  - [Emotion Dataset](#emotion-dataset)
  - [Song Dataset](#song-dataset)
- [Test Modules](#test-modules)
- [Data Pipeline](#data-pipeline)
- [Data Version Control (DVC)](#data-version-control-dvc)
- [Logging](#logging)
- [Data Schema and Statistics](#data-schema-and-statistics)
- [Anomaly Detection & Alerts](#anomaly-detection--alerts)
- [Pipeline Flow Optimization](#pipeline-flow-optimization)

## Features

- Real-time facial emotion detection
- Personalized music recommendations based on detected emotions
- Option to match current mood or change emotional state
- Continuous learning and adaptation to user preferences
- User feedback integration for improved recommendations

## Datasets

We primarily use the following datasets for our project:

- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) - Comprises images of human faces with emotion labels
- [Spotify Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) - Contains song metadata and audio features

## Team Members

- Anisha Kumari Kushwaha
- Anjith Prakash Chathan Kandy
- Francisco Chavez Gonzalez
- Jai Surya Kode
- Kirti Deepak Kshirsagar
- Nihira Golasangi

## Installation

This project uses `Python == 3.11`. Please ensure that the correct version is installed on your device. This project also works on Windows, Linux and Mac.

## Prerequisities

1. git
2. python==3.11
3. docker daemon/desktop is running

## User Installation

The steps for User installation are as follows:

1. Clone repository onto the local machine

```
git clone https://github.com/KodeJaiSurya/Vibify.git
```

2. Check python version == 3.11

```python
python --version
```

3. Check if you have enough memory

```docker
docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
```

<hr>

**FOR WINDOWS: Create a file called .env in the same folder as `docker-compose.yaml` and set the user as follows:**

```
AIRFLOW_UID=50000
```

**If you get the following error**

```
ValueError: Unable to configure handler 'processor'
```

**Setting the user manually like above fixes it**

<hr>

4. With Docker running, initialize the database. This step only has to be done once.

```docker
docker compose up airflow-init
```

5. Run airflow

```docker
docker-compose up
```

Wait until terminal outputs something similar to

`app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

6. Visit localhost:8080 login with credentials

```
user:airflow
password:airflow
```

7. Run the DAG by clicking on the play button on the right side of the window

8. Stop docker containers

```docker
docker compose down
```

## Tools Used for MLOps

- Airflow
  We use Apache Airflow to orchestrate our pipeline. We create a DAG with our modules.
- Docker
  We use Docker and containerization to ship our data pipeline with required dependencies installed making it platform independent. ‘docker-compose.yaml’ contains the code for running the Airflow.
- Github Actions
  We use Github Actions workflows to trigger the tests when any changes are merged or pushed onto the main branch. Currently Github Actions handles the triggering of DVC as well as Anomaly Detection.
- DVC
  Data version control helps to version the dataset ensuring reproducibility and traceability so that we can recreate any previous state of your project. It stores the metadata and stores the data in the cloud.
- Google Cloud Platform (GCP)
  The data version control is hosted on GCP. GCP helps to host large datasets and multiple people can access the data at once. It can be easily integrated with airflow.
- TensorFlow
  We use Tensorflow to validate the data schema and find any anomalies in the data.

## Data Preprocessing

### Emotion Dataset

1. **Data Loading and Verification**: Dataset downloaded from Google Drive and checked for availability.
2. **Data Loading in Chunks**: Dataset divided into smaller chunks for efficient processing.
3. **Emotion Filtering**: Data filtered to include 4 emotion labels (Happy, Sad, Angry, and Neutral).
4. **Emotion Mapping**: Filtered emotion labels mapped to standardized labels (0, 1, 2, 3).
5. **Pixel Preprocessing**: Pixel data converted to 2D arrays, reshaped, and normalized.
6. **Label Extraction**: Emotion labels separated into a target array.
7. **Data Persistence**: Processed images and label arrays saved as separate files.

### Song Dataset

1. **Data Loading and Verification**: Dataset downloaded from Google Drive and checked for availability.
2. **Data Cleaning**: Rows with missing values removed, duplicate entries eliminated.
3. **Feature Selection**: Relevant columns retained for analysis.
4. **Feature Scaling**: Numerical features scaled using StandardScaler.
5. **Data Storage**: Preprocessed data saved as a CSV file.

## Test Modules

- Unit tests implemented using the "unittest" framework.
- Tests cover various scenarios, including edge cases.
- Triggered by Github Actions workflow on push and pull requests.
- Can be run using the command line terminal.

## Data Pipeline

The data pipeline is modularized and coordinated using Apache Airflow. Key components include:

- `emotion_data_downloader.py`: Downloads emotion datasets.
- `emotion_data_processor.py`: Processes emotion images in chunks.
- `emotion_data_aggregator.py`: Combines processed chunks and saves final arrays.
- `emotion_data_pipeline.py`: Coordinates the complete emotion data workflow.
- `song_data_pipeline.py`: Handles end-to-end song data processing.

## Data Version Control (DVC)

DVC is used for data management and versioning:

1. **Setup DVC**: Initialized in the repository with Google Cloud Storage as remote.
2. **Download and Check Data**: Python scripts run to download and verify datasets.
3. **Pull and Push DVC Data**: Latest data pulled from remote, new data added and pushed.
4. **Commit DVC Changes**: DVC metadata committed to Git repository.

## Logging

Logging is implemented modularly in all data pipeline DAG functions, providing actionable messages for debugging and monitoring.

## Data Schema and Statistics

- TFDV used to generate statistics and infer schema for datasets.
- Statistics provide quantitative overview of data.
- Schema defines structure, types, and constraints for data features.

## Anomaly Detection & Alerts

- Anomalies detected using TFDV's `validate_statistics` against inferred schema.
- Visualization of anomalies provided by `tfdv.display_anomalies`.
- Custom `anomalies_info()` function implemented for alerts and error throwing.

## Pipeline Flow Optimization

- Initial execution time of 5 minutes reduced to 50 seconds.
- Optimization achieved through:
  - Parallel processing of independent pipeline parts.
  - Implementation of chunking strategy for larger dataset.
  - Improved failure handling and resource utilization.
