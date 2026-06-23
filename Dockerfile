FROM python:3.10-slim

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user specifically required for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user

# Copy the entire project and change ownership to the new user
COPY --chown=user:user . /app

# Switch to the new non-root user
USER user

# Make the startup script executable
RUN chmod +x start_prod.sh

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Set the entrypoint to the startup script
CMD ["./start_prod.sh"]
