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
2. Check python version  == 3.11
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
- Docker
- Github Actions
- DVC
- Google Cloud Platform (GCP)
- TensorFlow
