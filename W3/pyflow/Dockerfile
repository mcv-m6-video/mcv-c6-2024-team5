# Start from a Python base image which includes Python and pip
FROM python:3.8-slim

# Install system dependencies required for PyFlow and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the PyFlow repository
WORKDIR /app
RUN git clone https://github.com/pathak22/pyflow.git .

# Install Python dependencies including Pillow and OpenCV
RUN pip install numpy Cython Pillow opencv-python-headless

# Compile PyFlow
RUN python setup.py build_ext -i

# Copy kitti_run.py to the PyFlow directory
COPY kitti_run.py kitti_run.py
# Create a volume for the user to mount their data, it will be on /data
VOLUME /data

# Default command keeps the container running
CMD ["bash"]
