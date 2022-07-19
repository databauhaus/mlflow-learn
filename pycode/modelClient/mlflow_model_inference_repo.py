import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# model batch inference
# .. this loads a model from the MLflow repo

# (1) load the model from repo
model_name = "my-linear-model"
model_uri = f"models:/{model_name}/1"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# (2) model inference
d = {'actual': [0, 1, 2, 3]}
model_input = pd.DataFrame(data=d)
model_output = loaded_model.predict(model_input)

# (3) plot actual and predicted values
plt.scatter(model_output['actual'], model_output['predict'],  color='black')
plt.plot(model_output['actual'], model_output['predict'], color='blue', linewidth=3)
plt.show()