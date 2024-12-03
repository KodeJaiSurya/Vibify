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
4. Google Cloud Platform account with Vertex AI access

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
- MLFlow
  We use MLflow for managing models for Staging and Production as it allows us to reuse the models from artifacts registry and serve it.

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

### Test Modules

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

### Data Version Control (DVC)

DVC is used for data management and versioning:

1. **Setup DVC**: Initialized in the repository with Google Cloud Storage as remote.
2. **Download and Check Data**: Python scripts run to download and verify datasets.
3. **Pull and Push DVC Data**: Latest data pulled from remote, new data added and pushed.
4. **Commit DVC Changes**: DVC metadata committed to Git repository.

### Logging

Logging is implemented modularly in all data pipeline DAG functions, providing actionable messages for debugging and monitoring.

### Data Schema and Statistics

- TFDV used to generate statistics and infer schema for datasets.
- Statistics provide quantitative overview of data.
- Schema defines structure, types, and constraints for data features.

### Anomaly Detection & Alerts

- Anomalies detected using TFDV's `validate_statistics` against inferred schema.
- Visualization of anomalies provided by `tfdv.display_anomalies`.
- Custom `anomalies_info()` function implemented for alerts and error throwing.

### Pipeline Flow Optimization

- Initial execution time of 5 minutes reduced to 50 seconds.
- Optimization achieved through:
  - Parallel processing of independent pipeline parts.
  - Implementation of chunking strategy for larger dataset.
  - Improved failure handling and resource utilization.

## Model Pipeline Development
We have implemented our machine learning pipeline on Google Cloud Platform (GCP). We added our codebase, and we built images using Docker. Subsequently, we pushed the Docker images to the Artifact Registry. We then trained and served our model using Vertex AI.

- `emotion_model_pipeline.py`: Handles facial emotion model creation and training, including data loading, CNN architecture definition, model compilation, and training workflow.

- `song_model_pipeline.py`: Implements song mood clustering using PCA and KMeans, with functions for dimensionality reduction, cluster visualization, and mood assignment.

- `train.py`: Coordinates the complete model training workflow, managing both emotion and song models, including GCS data transfer, model saving, and result uploading.

### Overview
The model pipeline is built on Google Cloud Platform's Vertex AI and handles two main components:
- **Emotion Detection Model**: Processes facial expressions in real-time
- **Song Clustering**: Categorizes songs based on emotional attributes

### Architecture Components
- **Training Pipeline**: Manages model training on Vertex AI with MLflow tracking
- **Data Pipeline**: Handles data preprocessing and validation
- **CI/CD Pipeline**: Automates testing, deployment, and monitoring
- **Monitoring System**: Tracks model performance and triggers alerts

### CI/CD Setup
1. Automated Testing
   - Unit tests for model components
   - Integration tests for pipeline

2. Deployment Pipeline
   - Model validation checks
   - Automated rollback capability

3. Monitoring and Alerts
   - Performance metric tracking
   - Drift detection
   - Slack notifications for pipeline events

### Building and Pushing Docker Images

The following commands are used to build and push the emotion model training image to Google Container Registry (GCR).

### Prerequisites
Before running the build and push commands, ensure you have:
- Docker installed and running
- Google Cloud SDK configured
- Authentication configured for Google Container Registry

### Authentication Setup
```bash
gcloud config set project PROJECTID

gcloud auth configure-docker gcr.io
```

### Build and Push Commands
```bash
docker build -t gcr.io/PROJECT_ID/emotion-training:latest .

docker push gcr.io/PROJECT_ID/emotion-training:latest
```

### Model Monitoring

#### Metrics Tracked
- Model performance metrics
- Resource utilization
- Prediction latency
- Data drift indicators

#### Alert Channels
- Slack notifications from Github Actions
- Email alerts from Vertex AI
### DVC

Steps to initialize and track files using DVC

1. Initialize dvc in the parent directory of your local repository.
    ```python
    dvc remote add -d temp /tmp/dvcstore
    ```
2. Set up remote bucket.
    ```python
    dvc remote add -d temp /tmp/dvcstore
    ```
3. Add the location as default to your remote bucket.
    ```python
    dvc remote add -d myremote gs://<mybucket>/<path>
    ```
4. Don't forget to modify your credentials.
    ```python
    dvc remote modify --lab2 credentialpath <YOUR JSON TOKEN>```

### MLFlow

Most important declarations in the code:
1. Set your tracking uri for MLFlow.
    ```python
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    ```
2. Setting the base level for logging; only warnings and above (error,critical) shall be logged.
    ```python
    logging.basicConfig(level=logging.WARN)
    ```

3. Set up the logger.
    ```python
    logger = logging.getLogger(__name__)
    ```

4. Additionally, you may or may not choose to ignore warnings.
    ```python
    warnings.filterwarnings("ignore")
    ```
<hr>