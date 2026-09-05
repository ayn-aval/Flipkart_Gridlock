# Must be >= 3.11: pandas 3.x, numpy 2.4 and scikit-learn 1.9 all declare
# Requires-Python >=3.11, so the previous python:3.10-slim base could not install
# requirements.txt at all — the build failed on `pip install` before any code ran.
# 3.12 rather than 3.13/3.14 because torch, opencv and ultralytics have the longest
# track record there; every pin in requirements.txt has been resolved against
# manylinux x86_64 wheels for this exact version.
FROM python:3.12-slim

# Linux graphics libraries required for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user (UID 1000) as required by Hugging Face Spaces
RUN useradd -m -u 1000 user
COPY --chown=user:user . /app
USER user

RUN chmod +x start_prod.sh

# Build the processed dataset AND the models inside the image. Previously only
# forecasting.py ran here, so the container depended on data/processed/ artefacts
# being committed — including corridor_adjacency.csv, which nothing generated.
# Running both steps also avoids cross-version unpickling errors.
RUN python3 backend/data_cleaning.py && python3 backend/forecasting.py

EXPOSE 7860
CMD ["./start_prod.sh"]
