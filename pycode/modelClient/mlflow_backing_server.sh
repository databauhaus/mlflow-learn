# set our system MLflow tracking URI
# .. any code using the MLflow logging API will need this
export MLFLOW_TRACKING_URI=http://localhost:5000

# start the MLflow server
# .. includes repo - SQLite 
# .. includes UI - see tracking URI
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host localhost &
    
