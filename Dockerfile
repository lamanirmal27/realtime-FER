FROM python:3.11-slim

# Install system dependencies for dlib, OpenCV, and Pillow
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libopenblas-dev liblapack-dev \
    libx11-dev libgtk-3-dev \
    libboost-all-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose the port your app will run on
EXPOSE 8000

# Start the app using gunicorn (Flask WSGI server)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]