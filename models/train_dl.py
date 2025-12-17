import pandas as pd
import mlflow
import mlflow.h2o
import matplotlib.pyplot as plt
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

EXPERIMENT_NAME = "bike_sharing_exp"
MLFLOW_TRACKING_URI = "http://3.236.234.253:5000"

TRAIN = "splits/train.csv"
VALID = "splits/validate.csv"
TEST  = "splits/test.csv"

TARGET = "count"


def load_data():
    train = pd.read_csv(TRAIN)
    valid = pd.read_csv(VALID)
    test = pd.read_csv(TEST)

    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_valid, y_valid = valid.drop(columns=[TARGET]), valid[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def to_h2o(X, y):
    df = pd.concat([X, y], axis=1)
    return h2o.H2OFrame(df)


def eval_and_log(model, X, y, split_name):
    df = pd.concat([X, y], axis=1)
    frame = h2o.H2OFrame(df)

    preds = model.predict(frame).as_data_frame().iloc[:, 0].values
    actual = y.values

    rmse = mean_squared_error(actual, preds) ** 0.5
    mae = mean_absolute_error(actual, preds)
    r2 = r2_score(actual, preds)

    mlflow.log_metric(f"{split_name}_rmse", rmse)
    mlflow.log_metric(f"{split_name}_mae", mae)
    mlflow.log_metric(f"{split_name}_r2", r2)

    plt.figure()
    plt.scatter(preds, actual - preds, alpha=0.5)
    plt.xlabel("Predictions")
    plt.ylabel("Residuals")
    plt.title(f"Residuals ({split_name})")
    p = f"dl_residual_{split_name}.png"
    plt.savefig(p)
    mlflow.log_artifact(p)
    plt.close()

    plt.figure()
    plt.scatter(actual, preds, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted ({split_name})")
    p = f"dl_scatter_{split_name}.png"
    plt.savefig(p)
    mlflow.log_artifact(p)
    plt.close()


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    h2o.init(max_mem_size="6G")

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    train_h2o = to_h2o(X_train, y_train)
    valid_h2o = to_h2o(X_valid, y_valid)

    with mlflow.start_run(run_name="H2O_DeepLearning"):
        mlflow.set_tag("model_type", "deeplearning")
        model = H2ODeepLearningEstimator(
            hidden=[64, 32],
            epochs=20,
            activation="Rectifier",
            seed=42
        )

        mlflow.log_params({
            "hidden_layers": "[64, 32]",
            "epochs": 20,
            "activation": "Rectifier",
            "target": TARGET
        })

        model.train(
            x=X_train.columns.tolist(),
            y=TARGET,
            training_frame=train_h2o,
            validation_frame=valid_h2o
        )

        eval_and_log(model, X_valid, y_valid, "valid")
        eval_and_log(model, X_test, y_test, "test")

        mlflow.h2o.log_model(model, "model")

    h2o.shutdown(prompt=False)


if __name__ == "__main__":
    main()
