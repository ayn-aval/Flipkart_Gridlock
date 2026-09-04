FROM python:3.10-slim

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
