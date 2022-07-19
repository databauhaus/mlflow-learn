import pandas as pd
import pickle
import os
import mlflow

class MyLinearModel(mlflow.pyfunc.PythonModel):

    # simple linear model
    # y = mx + b

    # class variables
    model_file_name = 'model.pkl'

    # instantiate with model parameters
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept
    
    # equation
    def calculate(self, x: float) -> float:

        return self.slope * x + self.intercept
    
    # data set prediction
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        
        model_input['predict'] = self.calculate(model_input['actual'])
        
        return model_input
    
    # serialize model
    # .. we really don't need this, the parent class takes care of this
    def save(self, file_path: str):

        file_name = os.path.join(file_path, MyLinearModel.model_file_name)
        
        with open(file_name,'wb') as file:
            pickle.dump(self, file)

        pass
    
    # de-serialize model
    # .. we really don't need this, the parent class takes care of this
    @classmethod
    def load(self, file_path: str) :
        
        file_name = os.path.join(file_path, MyLinearModel.model_file_name)

        with open(file_name, 'rb') as file:
            loaded_model = pickle.load(file)

        return loaded_model


