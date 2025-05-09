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

In the `Makefile` we have `make format` that executes `poetry run black ./src/*.py`. This will check the code and make adjustments to make the formatting consistent.

## Type-checking:

For this the `mypi` library is used. In the `Makefile` we `make type-checking`. This run `poetry run mypy ./src/*.py`. This will check the `src` directory pythong files type hints

## Import sorting:

For this the `isort` library is used. In the `Makefile` the import sorting is executed via `make sort` this will run `poetry run isort ./src/*.py`.


## Linting:

For this the `pylint` library is used. In the `Makefile` this is performed via `make lint`. This will run `poetry run pylint --disable=R,C ./src/*.py`. The `--disable=R,C` means that the following is skipped. 

-   R: Refactor suggestions (e.g., "too many nested blocks").
-   C: Convention issues (e.g., naming style not matching PEP8).
