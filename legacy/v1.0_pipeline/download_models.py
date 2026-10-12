"""
Download the release v1.0 model weights and preprocessing artifacts from
Google Cloud Storage into the local folders run_pipeline.py expects. Run
this on any fresh machine (or at the start of an automated job) instead of
relying on files already being present.

Run:
    python download_models.py
"""

import os
from google.cloud import storage

BUCKET = "geopolitical-risk-forecasting-models"

FILES = [
    ("models/release_v1.0_goldstein.weights.h5", "model_release_v1.0_goldstein.weights.h5"),
    ("models/release_v1.0_direction.weights.h5", "model_release_v1.0_direction.weights.h5"),
    ("artifacts/goldstein_preprocessing.joblib", "artifacts/goldstein_preprocessing.joblib"),
    ("artifacts/direction_preprocessing.joblib", "artifacts/direction_preprocessing.joblib"),
]

client = storage.Client()
bucket = client.bucket(BUCKET)

for blob_name, local_path in FILES:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    bucket.blob(blob_name).download_to_filename(local_path)
    print(f"Downloaded gs://{BUCKET}/{blob_name} -> {local_path}")
