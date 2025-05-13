# Introduction 

A template for an ml project. 

Basic examples will be provided on tools that are used such as:

- poetry for dependency management
- developer tools like pylint/black/mypy
- mlflow for experiment tracking
- dvc for data versioning and pipeline

A Makefile to quickly execute repeated tasks.

Everything will be containerized.

## Poetry

Activate the poetry environment with `poetry shell`. 

You can add packages using `poetry add requests` this will add the `requests` packages with a version range to the `pyproject.toml`. You can also directly add packages to the `pyproject.toml` and then execute `poetry lock` to resolve the dependencies. Once finished you can execute `poetry install` to install the dependencies in the virtual environment. Flags like `--no-root` and `--sync` are telling poetry that we are not creating a packages and that dependencies not that are not in the `pyproject.toml` should be removed.

## Makefile

In python is mainly used to quickly execute commands that are often used. For example `make install` will execute `poetry lock` and then `poetry install`. Check the file for more commands in the Makefile.

## Formatter

In the `Makefile` we have `make format` that executes `poetry run black ./src/*.py`. This will check the code and make adjustments to make the formatting consistent.

## Type-checking:

For this the `mypi` library is used. In the `Makefile` we `make type-checking`. This run `poetry run mypy ./src/*.py`. This will check the `src` directory pythong files type hints

## Import sorting:

For this the `isort` library is used. In the `Makefile` the import sorting is executed via `make sort` this will run `poetry run isort ./src/*.py`.

## Linting:

For this the `pylint` library is used. In the `Makefile` this is performed via `make lint`. This will run `poetry run pylint --disable=R,C ./src/*.py`. The `--disable=R,C` means that the following is skipped. 

-   R: Refactor suggestions (e.g., "too many nested blocks").
-   C: Convention issues (e.g., naming style not matching PEP8).

## Github Actions

Github actions is used to trigger a CI testing workflow on a push to the main branch on github or on a pull request. It checks out the code, installs python, then poetry, the dependencies using poetry and finally runs a series of tests.

### Repo secrets

In case the workflow needs certains permissions, repository secrets can be used. These can be created under `settings` then `Secrets and Variables`. In the example github workflow it is shown how a secret can be used.

```
# This is a basic workflow to help you get started with Actions

name: CI

# Controls when the workflow will run
on:
  # Triggers the workflow on push or pull request events but only for the "main" branch
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# A workflow run is made up of one or more jobs that can run sequentially or in parallel
jobs:
  # This workflow contains a single job called "build"
  build:
    # The type of runner that the job will run on
    runs-on: ubuntu-latest

    # Steps represent a sequence of tasks that will be executed as part of the job
    steps:
      # Checks-out your repository under $GITHUB_WORKSPACE, so your job can access it
      - name: checkout code
        uses: actions/checkout@v4

      # install the python version we want to use. NOTE: this is a rather old python version
      - name: Set up Python 3.10.16
        uses: actions/setup-python@v4
        with:
          python-version: '3.10.16'
          
      # Install pipx 
      - name: Install pipx (needed for Poetry)
        run: python -m pip install --user pipx && python -m pipx ensurepath

      # Install poetry which we use for dependency management. 
      - name: Install Poetry
        run: pipx install poetry

      # Install the dependencies we need (e.g. pandas, scikit-learn)
      - name: Install dependencies
        run: |
          make install

      # Run the linter to make sure the code is PEP compliant
      - name: Lint with pylint
        run: |
          make lint

      # Format the code using black
      - name: Format with black
        run: |
          make format
          
      # Type check the code using mypy
      - name: Type check the code
        run: |
          make type-checking
          
      # Run unit tests with pytest 
      - name: Unit test with pytest
        run: |
          make tests

      # Runs a single command using the runners shell
      - name: list directory
        run: ls -la
```

## docker 

The `docker` directory has a Dockerfile that can be used to build and image and run a `train.py` and `process_data.py` script in a container. The directory also contains a `README.md` with more explenation on the files. Note that the `Dockerfile` contains comments for each commands that hopefully makes them sufficiently clear.  

The image can be build using `make build`. 

The train script can be run using `make train`

The process_data script can be run using `make process`

