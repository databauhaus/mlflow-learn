import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# model batch inference
# .. this loads a model from a file system artifact location

# (1) load the model in `python_function` format
model_uri = "mlflowRepo/my_linear_model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# (2) model inference
d = {'actual': [0, 1, 2, 3]}
model_input = pd.DataFrame(data=d)
model_output = loaded_model.predict(model_input)

# (3) plot actual and predicted values
plt.scatter(model_output['actual'], model_output['predict'],  color='black')
plt.plot(model_output['actual'], model_output['predict'], color='blue', linewidth=3)
plt.show()