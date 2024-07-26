# Drought-predictor-project (MLOps project)
![Drought](https://github.com/batxes/Drought-predictor-project/blob/main/image.jpg)

## About the project

This is an end-to-end Machine Learning project that trains a model with a pipeline, tracks the experiments, deploys the model, monitors the performance and follows best practices.

## Description of the project

We want to predict continental US drought levels using meteorological and soil data.

The data was downloaded from Kaggle and contains metereological and soil data.

For each day, we have different measurements and each day was classified with a score: None if there was no drought, and 1 to 5, if there was a drought, indicating:

- abnormally dry
- moderate drought
- severe drought
- extreme drought
- exceptional drought

The idea of the data is to predict or classify the level of drought there could be, using meteorological data, which can be forecasted, which then could also be predicted if there could be a drought or not.

Therefore, We trained a machine learning model that predicts if there could be a drought or not.

The dataset can be found in: https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data

## About the model

The model is trained with Gradient Boosting algorithm.
The dataset was last updated 3 years ago.

## About the project

- Data was explored, cleaned and preprocessed first in a jupyter notebook. I trained also the model in the notebook.
- The code was cleaned and exported to a python script. 
- I track the experiments and register the model using Mlflow.
- I used Prefect for workflow orchestration.

### Steps I followed
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

