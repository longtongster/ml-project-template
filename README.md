# Introduction 

An example repo that uses poetry, developer tools like pylint/black/mypy but also mlflow and dvc.

The repo will use a Makeile to quickly execute repeated tasks.

Everything will be containerized

## Poetry

Activate the poetry environment with `poetry shell`. 

You can add packages using `poetry add requests` this will add the `requests` packages with a version range to the `pyproject.toml`. You can also directly add packages to the `pyproject.toml` and then execute `poetry lock` to resolve the dependencies. Once finished you can execute `poetry install` to install the dependencies in the virtual environment. Flags like `--no-root` and `--sync` are telling poetry that we are not creating a packages and that dependencies not that are not in the `pyproject.toml` should be removed.

## Makefile

In python is mainly used to quickly execute commands that are often used. For example `make install` will execute `poetry lock` and then `poetry install`. Check the file for more commands in the Makefile.

## Formatter

In the `Makefile` we have `make format` that executes