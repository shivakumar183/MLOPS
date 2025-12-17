import mlflow
import mlflow.h2o
import h2o
from fastapi import FastAPI
from datetime import datetime

from fastapi_app.schemas import BikeRecord, BatchRequest
from fastapi_app.utils import convert_to_h2o, convert_batch_to_h2o
from fastapi_app.config import (
    MLFLOW_TRACKING_URI,
    DL_MODEL_URI,
    GBM_MODEL_URI,
    GLM_MODEL_URI,
)

h2o.init() 

app = FastAPI(
    title="Bike Sharing Model API",
    description="Serving DL, GBM, and GLM models from MLflow Model Registry",
    version="1.0",
)

@app.on_event("startup")
def startup_event():
    global dl_model, gbm_model, glm_model

    print("Setting MLflow tracking URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    dl_model = mlflow.h2o.load_model(DL_MODEL_URI)
    gbm_model = mlflow.h2o.load_model(GBM_MODEL_URI)
    glm_model = mlflow.h2o.load_model(GLM_MODEL_URI)
    print("All models loaded")


@app.post("/predict_dl")
def predict_dl(record: BikeRecord):
    try:
        hf = convert_to_h2o(record.dict())
        pred = dl_model.predict(hf).as_data_frame()["predict"][0]
        return {
            "model": "DL_model",
            "prediction": float(pred),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_gbm")
def predict_gbm(record: BikeRecord):
    try:
        hf = convert_to_h2o(record.dict())
        pred = gbm_model.predict(hf).as_data_frame()["predict"][0]
        return {
            "model": "GBM_model",
            "prediction": float(pred),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_glm")
def predict_glm(record: BikeRecord):
    try:
        hf = convert_to_h2o(record.dict())
        pred = glm_model.predict(hf).as_data_frame()["predict"][0]
        return {
            "model": "GLM_model",
            "prediction": float(pred),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_batch")
def predict_batch(payload: BatchRequest):
    try:
        hf = convert_batch_to_h2o([r.dict() for r in payload.records])
        dl_preds = dl_model.predict(hf).as_data_frame()["predict"].tolist()
        gbm_preds = gbm_model.predict(hf).as_data_frame()["predict"].tolist()
        glm_preds = glm_model.predict(hf).as_data_frame()["predict"].tolist()

        return {
            "count": len(payload.records),
            "dl_predictions": dl_preds,
            "gbm_predictions": gbm_preds,
            "glm_predictions": glm_preds,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}
