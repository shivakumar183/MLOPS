import h2o
from h2o.automl import H2OAutoML
from pathlib import Path

TRAIN = "splits/train.csv"
VALID = "splits/validate.csv"

TARGET = "count"

MODEL_SAVE_DIR = Path("automl_models")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

RUNTIME = 120

def main():

    print("Initializing H2O")
    h2o.init(max_mem_size="4G")

    print("Loading datasets")
    train = h2o.import_file(TRAIN)
    valid = h2o.import_file(VALID)

    x = [col for col in train.columns if col != TARGET]

    print("Running AutoML...")
    aml = H2OAutoML(
        max_runtime_secs=RUNTIME,
        nfolds=5,
        sort_metric="RMSE",
        seed=42,
    )
    aml.train(x=x, y=TARGET, training_frame=train, validation_frame=valid)

    lb = aml.leaderboard
    print(lb)

    lb_df = lb.as_data_frame()
    leaderboard_path = MODEL_SAVE_DIR/"leaderboard.csv"
    lb_df.to_csv(leaderboard_path, index=False)
    print(f"\nSaved leaderboard → {leaderboard_path}")

    # Getting top 3 model IDs
    model_ids = lb_df["model_id"].iloc[:3].tolist()
    print("\nTop 3 Models:", model_ids)

    saved_models = []

    print("\nSaving top 3 models")
    for model_id in model_ids:
        model = h2o.get_model(model_id)
        path = h2o.save_model(model=model, path=str(MODEL_SAVE_DIR), force=True)
        saved_models.append(path)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
