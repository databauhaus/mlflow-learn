# MLflow quickstart
# https://www.mlflow.org/docs/latest/models.html
# https://www.mlflow.org/docs/latest/models.html#model-customization

# this example saves a model as a standalone artifact
# .. provide a file system for persistence

import os
import shutil
import mlflow
from models import MyLinearModel

# construct and save the model
# .. replace if exists
model_uri = "mlflowRepo/my_linear_model"

try:
    shutil.rmtree(model_uri, ignore_errors=False, onerror=None)
except FileNotFoundError as ex:
    print (f'NOTE: No exising model found [{model_uri}]')

# instantiate and save the model in pyfunc flavor
my_model = MyLinearModel(slope=2, intercept=1)
mlflow.pyfunc.save_model(path=model_uri, python_model=my_model)