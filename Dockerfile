# EV Fleet Intelligence Brain — container image.
# Builds data + model at image build time so the container starts demo-ready.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Install dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App source.
COPY . .

# Pre-build synthetic data + train the battery model so first load is instant.
RUN python run_pipeline.py

EXPOSE 8501 8000

# Default: the Streamlit dashboard. Override the command to run the API instead:
#   docker run ... uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
