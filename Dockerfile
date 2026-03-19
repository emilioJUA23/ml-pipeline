FROM python:3.12.4-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY red-wine-quality.csv .

# Directories that will be mounted as volumes in production;
# create them so the container works standalone too
RUN mkdir -p data mlruns mlflow-artifacts

# Suppress MLflow's noisy git-not-found warning
ENV GIT_PYTHON_REFRESH=quiet

# MLflow tracking server port
EXPOSE 5000

# Default: run training with configurable hyperparams
# Override CMD at runtime, e.g.: docker run ... python main.py --alpha 0.3
CMD ["python", "main.py"]
