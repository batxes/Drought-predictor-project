# Drought-predictor-project



### steps I followed
1- pipenv install -r requirements.txt
2- pipenv shell
3- Create data, models, script folders and Dockerfile, docker-compose.yml, Makefile for later.
4- jupyter notebook -> Model training
5- Check the variables in mlflow with: pipenv run mlflow ui
6- Create a docker-compose.yml file with the MLflow tracking server.
7- docker-compose up (check the the mlflow ui is working)
8- Take the code in the notebook to train.py
9- Add prefect. Add to requirements.txt and also install with "pipenv install prefect"
10- Define functions in the train.py: load, clean, train, promote...
11- Run the flow with: python train.py
12- Check it with: prefect server start -> http://127.0.0.1:4200/dashboard
13- I wanted to add the prefect commands to the docker compose but I am getting errors.
14-

