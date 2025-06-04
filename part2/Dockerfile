# Stage 1: build dependencies
FROM python:3.9-slim 

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


WORKDIR /app

COPY . /app/


CMD ["uvicorn", "drift_detector:app", "--host", "0.0.0.0", "--port", "8080"]
