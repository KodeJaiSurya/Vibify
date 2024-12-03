FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY task.py .

ENTRYPOINT ["python", "task.py"]