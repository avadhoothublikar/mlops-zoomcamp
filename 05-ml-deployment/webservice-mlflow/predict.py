import pickle
import os
import mlflow

from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient


RUN_ID = os.getenv('RUN_ID')  # Replace with your actual run ID
#MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#logged_model = f"runs:/{RUN_ID}/model"

logged_model = f"s3://mlflow-model-avad/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


# client = MlflowClient(MLFLOW_TRACKING_URI)
# model_version = None
# for mv in client.search_model_versions(f"run_id='{RUN_ID}'"):
#     model_version = mv.version
#     break

def prepare_features(ride):
    features = {}
    features["PU_DO"] = '%s_%s' % (ride["PULocationID"], ride["DOLocationID"])  
    features["trip_distance"] = ride["trip_distance"]
    return features

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        "duration": pred,
        "model_version": RUN_ID
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
