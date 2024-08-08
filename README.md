# Drought-predictor-project (MLOps project)
![Drought](https://github.com/batxes/Drought-predictor-project/blob/main/image.jpg)

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

## About the project

- Data was explored, cleaned and preprocessed first in a jupyter notebook. First model training experiments were done in the notebook.
- The code was cleaned and exported to a python script.
- I tracked the experiments and register the model using Mlflow.
- I used Prefect for workflow orchestration.
- The model is containerized and can be deployed with docker. It starts a Gunicorn server with the prediction script which uses Flask.
- The model can be tested with send_prediction_request.py

## How to run the project

- clone the project
- mkdir data
- Download the dataset from https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data?resource=download/ into the data folder
- In the project directory, run `pipenv install -r requirements` and then `pipenv shell`
- run Mlflow with `docker compose up` -> You can check the experiments information in http://localhost:5001/
- run the project with `python train.py` -> You can check the pipeline in http://localhost:4200/
- docker build -t drought_predictor .
- docker run -d -p 5002:5002 --name drought_prediction drough_predictor  -> to see that it is running
- run now: python send_prediction_request.py

### Steps I followed
1. pipenv install -r requirements.txt
2. pipenv shell
3. Create data, models, script folders and Dockerfile, docker-compose.yml, Makefile for later.
4. jupyter notebook -> Model training
5. Check the variables in mlflow with: pipenv run mlflow ui
6. Create a docker-compose.yml file with the MLflow tracking server.
7. docker compose up (check the the mlflow ui is working)
8. Take the code in the notebook to train.py
9. Add prefect. Add to requirements.txt and also install with "pipenv install prefect"
10. Define functions in the train.py: load, clean, train, promote...
11. Run the flow with: python train.py
12. Check it with: prefect server start -> http://127.0.0.1:4200/dashboard
13. I wanted to add the prefect commands to the docker compose but I am getting errors.
14. I generated a predict script that generates a random data and gets the prediction (predict.py)
15. Convert predict.py into Flask app, so we can run from a docker container.
16. Create another script that sends the request. Add the Data_generation function to it.
17. Modify the Dockerfile
18. Install flask and gunicorn
19. docker build -t drought_predictor .
20. docker run -d -p 5002:5002 --name drought_prediction drough_predictor  -> to see that it is running
21. run now: python send_prediction_request.py
22. I added some metrics to be recorded in the training and in the prediction scripts with prometheus, which will be then followed using grafana. docker-compose was updated and I added prometheus.yml file
23. Grafana -> http://localhost:3000/
24. Prometheus -> http://localhost:9090/
24. Recorded metrics -> http://localhost:5002/metrics
25. Added some unit tests and integration test
25. I am setting flake8, a code linter. pipenv isntall flake8. I added the .flake8 file and it can be run typing: flake8
26. I also installed black code formater. run: black .
27. I will use pre-commit to automatically run flake8 and black before commits. Create .pre-commit-config.yaml. Install it: pre-commit install
28. Add a CI/CD pipeline. Create .github/workflows/ci_cd.yml
29. Create a Makefile to run everything
