# Docker Setup for ML Project

This directory contains Docker-related files for containerizing the ML project.

## Files

- `Dockerfile`: Main Dockerfile for building the ML project image
- `.dockerignore`: Specifies files to exclude from the Docker build context
- `docker-compose.yml`: Docker Compose configuration for running the containerized application

## Usage

### Building the Docker Image

You can build the Docker image using the Makefile from the project root:

```bash
make build
```

This will build the image using the Dockerfile in this directory.

### Running the Container

To run the container:

```bash
make run
```

This will start the container in detached mode and expose port 8080.

### Stopping the Container

To stop the running container:

```bash
make stop
```

### Using Docker Compose

Alternatively, you can use Docker Compose:

```bash
# From the docker directory
docker-compose up -d

# To stop
docker-compose down
```

## Configuration

The Docker setup mounts the following directories from the host to the container for data persistence:

- `raw_data`: Contains raw input data
- `processed_data`: Contains processed data
- `artifacts`: Contains model artifacts like preprocessors
- `saved_models`: Contains trained models

## Customization

You can modify the Dockerfile or docker-compose.yml to:

- Change the Python version
- Add additional dependencies
- Modify environment variables
- Change exposed ports
- Adjust volume mounts