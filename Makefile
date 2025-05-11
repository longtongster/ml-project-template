install:
	# install commands - first activate with `poetry shell`
	poetry lock &&\
	poetry install --no-root --sync

format:
	# format code
	poetry run black ./src/*.py # to add a directory use `mylib/*.py`

format-diff:
	# black executes a dry run to show changes in code (not changing the files though)
	poetry run black --diff *.py

type-checking:
	# mypy executes type checking
	poetry run mypy ./src/*.py

sort:
	poetry run isort ./src/*.py

lint:
	# flake8 or pylint
	# poetry run pylint --disable=R,C *.py # mylib/*.py
	poetry run pylint --disable=R,C ./src/*.py

tests:
	# test
	poetry run pytest -vv --cov=src ./tests/*.py

process-data:
	poetry run python ./src/process_data.py

build:
	# build docker image
	docker build -t wiki-app-image .

run: build
	# run a docker container from the wiki-app image
	docker run --name wiki-app-container --rm -p 8080:8080 -d wiki-app-image

stop: 
	# stop the container from running
	docker stop wiki-app-container 

deploy:
	# push the image to ECR
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 891376951737.dkr.ecr.us-east-1.amazonaws.com
	docker build -t wikipedia/wiki-app .
	docker tag wikipedia/wiki-app:latest 891376951737.dkr.ecr.us-east-1.amazonaws.com/wikipedia/wiki-app:latest
	docker push 891376951737.dkr.ecr.us-east-1.amazonaws.com/wikipedia/wiki-app:latest

all: install lint test deploy