# Start with a Python 3.10 slim image as the base
# The slim variant is smaller than the standard image but still includes necessary libraries
FROM python:3.10-slim

# Set the working directory inside the container
# All subsequent commands will be run from this directory
WORKDIR /app

# Install system dependencies required for building Python packages
# build-essential includes gcc, g++, and make which are needed for compiling some Python packages
# --no-install-recommends reduces the image size by not installing recommended but non-essential packages
# The last command removes the apt cache to reduce image size
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry - the Python dependency management tool
# --no-cache-dir prevents pip from storing package archives locally
# Specifying the version ensures reproducibility
RUN pip install --no-cache-dir poetry==1.8.3

# Copy only the dependency definition files first
# This is a Docker best practice that allows Docker to cache the dependency installation
# step if the dependencies haven't changed, even if other source code has
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create a virtual environment inside the container
# This is because the container itself provides isolation, so a virtual env is redundant
RUN poetry config virtualenvs.create false

# Install project dependencies
# --no-root: Don't install the project package itself
# --without dev: Skip development dependencies to keep the image smaller
# --no-interaction: Don't ask any interactive questions
RUN poetry install --no-root --without dev --no-interaction

# Create necessary directories if they don't exist
# These directories are used for storing data, models, and artifacts
RUN mkdir -p ./artifacts ./processed_data ./saved_models

# Create a non-root user and group for security
# -r: Create a system account
# groupadd: Create a new group called 'appuser'
# useradd: Create a new user called 'appuser' and:
#   -r: Make it a system account (no aging information, no home directory by default)
#   -g appuser: Add the user to the 'appuser' group
#   -m: Create a home directory for the user
#   -d /home/appuser: Specify the home directory location
# This follows security best practices by not running the container as root
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy the rest of the application code into the container
# This happens after dependency installation to leverage Docker's layer caching
COPY . .

# Set ownership of all application files to the non-root user
# -R: Apply recursively to all files and directories
# This ensures the non-root user has proper permissions to access and modify files
RUN chown -R appuser:appuser /app

# Set environment variables
# PYTHONPATH ensures Python can find modules in the /app directory
# PYTHONUNBUFFERED ensures Python output is sent straight to the terminal without buffering
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV SCRIPT_TO_RUN=main.py

# Switch to the non-root user for all subsequent commands
# This improves security by not running the application as root
USER appuser

# Define the command that will run when the container starts
# This uses a shell to expand the environment variable
CMD ["sh", "-c", "python ./src/${SCRIPT_TO_RUN}"]

# Uncomment to expose a port if your application has a web interface
# This maps a port from the container to the host
# EXPOSE 8080