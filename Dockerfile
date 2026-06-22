FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Make the startup script executable
RUN chmod +x start_prod.sh

# Expose port 8000
EXPOSE 8000

# Set the entrypoint to the startup script
CMD ["./start_prod.sh"]
