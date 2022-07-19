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

# this example creates an experiment and tracks it in the configured location
# .. in this case the MLflow repo plus artifacts location

import mlflow

# (1) set tracking location
# .. this is where we will track experiments and artifacts in a filestore
#tracking_uri = "file:///Users/andreasmarx/Projects/DataBauhaus/CIBC/PBB/development/pycode/mlruns"

# .. this is where we will track the experiments in a repo
# .. not required if MLFLOW_TRACKING_URI has been set
# tracking_uri = "http://localhost:5000"
#mlflow.set_tracking_uri(tracking_uri)

print(f"Current tracking uri: {mlflow.get_tracking_uri()}.")

# (2) create experiment
# .. the experiment name must be unique and case sensitive
# .. we are using a uuid to create a new experiment with each execution

try:
    experiment_id = mlflow.create_experiment("MyLinearModel Experiments")
    experiment = mlflow.get_experiment(experiment_id)

    # write out some experiment attributes
    print(f"Name: {experiment.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tags: {experiment.tags}")
    print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

except Exception as e:
    print("ERROR: An error occured creating the experiment. It may already exist.")
    print(e)



