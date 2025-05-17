# TODO

- docker-compose - setup permissions for mounted directories maybe for now via docker compose later via the makefile
- use logger instead of print statements.
- adjust the train.py script such that it takes the processed data as input and creates several outputs

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

### Create a workflow file

In your reposistory click on `Actions` then type `simple` to get a `simple workflow` can be selected. This can be adjusted to your specific needs.

### Repo secrets

In case the workflow needs certains permissions, repository secrets can be used. These can be created under `settings` then `Secrets and Variables`. In the example github workflow it is shown how a secret can be used.

### predefined uses

There are quite some different features that can be used in github actions. such as checkout the repo using `actions/checkout@v4`. There are also features for continuous machine learning (cml) and dvc (`iterative/setup-dvc@v3`). After that you can `run: dvc repro`. There are quite some advanced examples (such as rendering plots in comments) in the `references` directory slides.

### Environment variables

The workflow file contains and example or a global and job variable

### Workflow file

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

## DVC

DVC in general is used data versioning and pipelines. For experimentation MLFlow is normally used.

In order to use DVC we need to have a git repository and then initalize a dvc repository using `dvc init`.

To see what is tracked by dvc you can execute `dvc list . --dvc-only`

### DVC pipelines

The dvc pipeline is defined in a `dvc.yaml` file.

To run the pipeline execture `dvc repro`.If nothing changed in certains `stages` in the pipeline DVC will not execute them. A full calculation of the pipeline can be forces usig `dvc repro --force".

You can see the DAG by entering `dvc dag`

**important** - the dvc pipeline only tracks what is mentioned under `outs` in the dvc.yaml file. The other files e.g. `.py` files, dvc.yaml, dvc.lock, `.dvc` are being tracked by git. With `dvc push` all files in the `outs` section are pushed to the remote data repo. This means by default the `raw_data` (not in outs) directory is not tracked by dvc. 

To run a specific stage from the `dvc.yaml` e.g. `preprocess` one can execute `dvc repro preprocess`.

You can also define a `metrics` argument that refers to a metrics file `metrics.json`. With `dvc metrics show` you can see the metrics. You can then change a hyperparameter, rerun the pipeline and use `dvc metrics diff` to see the difference in metrics.

### DVC Data versioning

In preperation of storing data in a remoe s3 location it is recommended to start an AWs sandbox. Then create an IAM user with S3 fullaccess that has programmatic access.
Then create a bucket `dvc-bucket-svw-1`. Then use `aws configure` to create the credentials locally. Test the setup with `aws s3 ls` which should return the created bucket. 

To add a datafile to be tracked by dvc execute `dvc add raw_data/ams_unprocessed_data.cvs`.

Then to track the versioning with dvc ` git add raw_data/ames_unprocessed_data.csv.dvc raw_data/.gitignore`

You can see what is tracked using  `cat raw_data/ames_unprocessed_data.csv.dvc`

A remote is folder or cloud provider where the actual data is stored. A remote can be created using

`dvc remote add AWSremote s3://dvc-bucket-svw-1`

This results in an entry in the `.dvc/config` file:

```
['remote "awsremote"']
    url = s3://dvc-bucket-svw-1
```

with `dvc remote list` you can list the available remotes.

local remotes can also be setup for rapid prototyping 

`dvc remote add --local myLocalremote /tmp/dvc`

You can also set the a default repo by using `-d` e.g., `dvc remote add -d AWSremote s3://dvc-bucket-svw-1`. 
With the default set commands such `dvc push` will use the default remote.

Use the `-r` flag as to push to a different location than the default one:

`dvc push -r myAWSremote data.csv`

### Tracking Data Changes

Below find the workflow for when you change the data.

`dvc add /path/to/datafile (e.g. data.csv)`

commit changed `.dvc` file to git

`git add path/to/datafile.dvc`
`git commit path/to/datafile.dvc -m "dataset updates"`

push metadata changes to Git

`git push origin main`

push changed data

`dvc push`