# README: Running Coarse2Fine Optical Flow with PyFlow

This guide provides instructions on how to run the Coarse2Fine Optical Flow computation using the PyFlow library. The script is designed to work with grayscale images, calculating the optical flow between two frames and generating both the flow visualization and a warped version of the second image based on the computed flow.

## Prerequisites

- **Docker**: Ensure you have Docker installed and running on your system. This project uses Docker to simplify dependency management and execution environment.
- **Images**: You'll need two grayscale images to compute the optical flow. Place these images in a directory named `local_data` at the root of your project directory.

## Building the Docker Container

First, you need to build the Docker container which encapsulates the PyFlow environment and dependencies:

1. Navigate to the root of the project directory where the `Dockerfile` is located.
2. Build the Docker image with the following command:

    ```bash
    docker build -t pyflow .
    ```

    This command creates a Docker image named `pyflow`.

## Preparing Your Data

Place your two grayscale images in a directory named `local_data` at the root of your project directory. These images should be named in a way that indicates their order, for example, `000045_10.png` and `000045_11.png`.

## Running the Script

To run the optical flow computation:

1. Use the following Docker command to start the container and mount the `local_data` directory to the `/data` directory inside the container:

    ```bash
    docker run -it -v $(pwd)/local_data:/app/data pyflow
    ```

2. Within the Docker container, run the script with the following command format:

    ```bash
    python script_name.py --image1 /app/data/first_image.png --image2 /app/data/second_image.png --output_flow /app/data/flow.png --output_warped /app/data/warped.png --viz
    ```

    - Replace `script_name.py` with the name of your Python script.
    - Replace `first_image.png` and `second_image.png` with the actual names of your images.
    - The `--viz` flag is optional. If provided, the script will generate and save the flow visualization and the warped second image in the specified paths.
    
    For example:

    ```bash
    python kitti_run.py --image1 /app/data/000045_10.png --image2 /app/data/000045_11.png --output_flow /app/data/flow.png --output_warped /app/data/warped.png --viz
    ```

## Outputs

After running the script, you'll find the following outputs in your `local_data` directory:

- `flow.png`: A visualization of the optical flow computed between the two input images.
- `warped.png`: The second image warped according to the computed optical flow, aligning it with the first image.

This README provides a basic overview of running a Coarse2Fine Optical Flow computation using PyFlow in a Docker environment. Adjust paths and filenames as necessary to fit your project structure and data.
