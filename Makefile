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
	docker build -t ml-project-image -f docker/Dockerfile .

process: build
	# Run the process_data.py in the container
	docker run --name process_data_container -e SCRIPT_TO_RUN=process_data.py ml-project-image

train: build
	# run the train script
	docker run -e SCRIPT_TO_RUN=train.py ml-project-image

stop: 
	# stop the container from running
	docker stop ml-project-container 

deploy:
	# push the image to ECR
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 891376951737.dkr.ecr.us-east-1.amazonaws.com
	docker build -t ml-project/ml-app -f docker/Dockerfile .
	docker tag ml-project/ml-app:latest 891376951737.dkr.ecr.us-east-1.amazonaws.com/ml-project/ml-app:latest
	docker push 891376951737.dkr.ecr.us-east-1.amazonaws.com/ml-project/ml-app:latest

docker-compose-up:
	# run using docker-compose
	cd docker && docker-compose up -d

docker-compose-down:
	# stop and remove containers created by docker-compose
	cd docker && docker-compose down

all: install lint test deploy