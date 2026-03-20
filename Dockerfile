FROM python:3.12.4-slim

WORKDIR /app

# Install dependencies first (layer cache) — code comes in via volume mount
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create runtime directories so the container works before volumes are mounted
RUN mkdir -p data/raw mlruns mlflow-artifacts

# Suppress MLflow's noisy git-not-found warning
ENV GIT_PYTHON_REFRESH=quiet

# MLflow UI port
EXPOSE 5000

CMD ["python", "main.py"]
