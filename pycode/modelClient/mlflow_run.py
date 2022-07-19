# MLflow quickstart
# https://www.mlflow.org/docs/latest/tracking.html#concepts
# https://www.mlflow.org/docs/latest/quickstart.html

"""
By default, the MLflow Python API logs runs locally to files in an mlruns directory 
wherever you ran your program. You can then run mlflow ui to see the logged runs.

There are different kinds of remote tracking URIs:

- Local file path (specified as file:/my/local/dir), where data is just directly stored locally.
- we skip database, HTTP service and Databricks workspace for now

For storing runs and artifacts, MLflow uses two components for storage: backend store 
and artifact store. While the backend store persists MLflow entities (runs, parameters, 
metrics, tags, notes, metadata, etc), the artifact store persists artifacts (files, models, 
images, in-memory objects, or model summary, etc).

This example explores using MLflow on the local machine. 
Backend and artifact store share a directory on the local filesystem—./mlruns

In this scenario, the MLflow client uses the following interfaces to record MLflow entities 
and artifacts:
- An instance of a LocalArtifactRepository (to store artifacts)
- An instance of a FileStore (to save MLflow entities)

"""

# experiment run
# .. creates and MLflow run under a given experiment
# .. instantiates the model
# .. log parameters
# .. log model as artifact
# .. log test data as artifact
# .. log plot of test data as artifact

import os
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from models import MyLinearModel

# for storage of temporary artifacts
temp_dir = "./temp"

# (1) start a run
# .. connect the run with the experiment created earlier
experiment_id = '1'

with mlflow.start_run(experiment_id=experiment_id, run_name='Model Test') as active_run:

    # (2) set and log model parameters
    slope = 2
    intercept = 1

    mlflow.log_param("slope", slope)
    mlflow.log_param("intercept", intercept)

    # (3) instantiate model
    # .. also log model as artifact
    # .. also regeister model in the repo
    # .. each repo registration also creates a new model version so be careful here
    my_model = MyLinearModel(slope=slope, intercept=intercept)
    model_info = mlflow.pyfunc.log_model(artifact_path="my-linear-model", 
            registered_model_name="my-linear-model", 
            python_model=my_model)

    # (4) load the model to make it a PyFuncModel
    # .. can't do class casting in python
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # (5) create test data
    # .. create an input dataframe
    d = {'actual': [0, 1, 2, 3]}
    model_input = pd.DataFrame(data=d)

    # (6) predict
    model_output = loaded_model.predict(model_input)

    # (7) save model output as artifact
    model_output.to_csv(f'{temp_dir}/model_test_output.csv', index=False)
    mlflow.log_artifact(f'{temp_dir}/model_test_output.csv')

    # (8) plot and save as artifact
    plt.scatter(model_output['actual'], model_output['predict'],  color='black')
    plt.plot(model_output['actual'], model_output['predict'], color='blue', linewidth=3)
    plt.savefig(f'{temp_dir}/model_test_output.png')
    mlflow.log_artifact(f'{temp_dir}/model_test_output.png')
    
    # (8) cleanup residuals
    try:
        os.remove(f'{temp_dir}/model_test_output.csv')
        os.remove(f'{temp_dir}/model_test_output.png')
    except Exception as e:
        print(f"ERROR: An error occurred during residual cleanup in [{temp_dir}]")