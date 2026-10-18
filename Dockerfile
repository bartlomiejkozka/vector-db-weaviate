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

# Set container environment flag and add /app to PATH
ENV IN_DOCKER=true \
    PATH="/app:${PATH}"

# Ensure LF line endings and executable permissions for the db script
RUN sed -i 's/\r$//' /app/db && chmod +x /app/db

CMD ["tail", "-f", "/dev/null"]
