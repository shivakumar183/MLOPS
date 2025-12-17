import mlflow
mlflow.set_tracking_uri("http://44.200.140.111:5000")

client = mlflow.MlflowClient()
client.delete_model_version(
    name="DL_model",
    version=2
)
