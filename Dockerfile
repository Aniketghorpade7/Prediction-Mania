# Use a lightweight, stable official Python runtime image
FROM python:3.11-slim

# Install light system compilation utilities needed for specialized packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the clean local timesfm source package directory structure first
COPY timesfm /app/timesfm

# Install the custom timesfm framework package in editable mode directly inside the container
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e /app/timesfm

# Install huggingface_hub CLI tools
RUN pip install huggingface_hub mpld3

# Pre-download TimesFM weights at BUILD time
RUN --mount=type=secret,id=hf_token \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='google/timesfm-2.5-200m-pytorch', \
    token=open('/run/secrets/hf_token').read().strip())"

# Copy the dashboard management workspace files 
COPY dashboard /app/dashboard

# Install the final dashboard interface application requirements
RUN pip install --no-cache-dir -r /app/dashboard/requirements.txt

# Explicitly link directory paths so the runtime environment never loses track of modules
ENV PYTHONPATH="/app/timesfm/src:/app/dashboard:${PYTHONPATH}"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
