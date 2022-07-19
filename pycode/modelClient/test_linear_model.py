import pandas as pd
import matplotlib.pyplot as plt
from models import MyLinearModel

# test the model equation
# .. instantiate new model
my_model = MyLinearModel(slope=2.0, intercept=1.0)

x = 1.0
y = my_model.calculate(x)
print (f'Test output for [{x}] is [{y}]')

# test the model prediction
# .. create an input dataframe
d = {'actual': [0, 1, 2, 3]}
model_input = pd.DataFrame(data=d)

# .. predict
# .. ignore the context, only needed for MLflow
model_output = my_model.predict(context=None, model_input=model_input)

# .. plot actual and predicted values
plt.scatter(model_output['actual'], model_output['predict'],  color='black')
plt.plot(model_output['actual'], model_output['predict'], color='blue', linewidth=3)
plt.show()

# test model serialization and de-serialization
file_store_path = '/Users/andreasmarx/Projects/DataBauhaus/CIBC/PBB/development/pycode/filestore'

# .. save model as artifact
my_model.save(file_store_path)

# model batch inference
loaded_model = MyLinearModel.load(file_store_path)
d2 = {'actual': [4, 5, 6, 7]}
model_test_input = pd.DataFrame(data = d2)

model_test_output = loaded_model.predict(context=None, model_input=model_test_input)
print (model_test_output)

plt.scatter(model_test_output['actual'], model_test_output['predict'],  color='black')
plt.plot(model_test_output['actual'], model_test_output['predict'], color='green', linewidth=3)
plt.show()
