FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training code
COPY emotion_model_pipeline.py .
COPY song_model_pipeline.py .
COPY train.py .

# Set the entrypoint
ENTRYPOINT ["python", "train.py"]