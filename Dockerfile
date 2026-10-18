FROM python:3.12-slim

WORKDIR /app

# Install system deps for h5py
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make the db script executable
RUN chmod +x /app/db

CMD ["tail", "-f", "/dev/null"]
