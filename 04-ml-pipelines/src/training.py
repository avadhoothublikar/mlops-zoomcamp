import pickle
import tempfile
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import os
import mlflow
import mlflow.xgboost

#configuration
mlflow_tracking_uri = "http://mlflow:5000"
experiment_name = "nyc-taxi-experiment"


#Data Loading
def read_dataframe(year, month):
    """Download and preprocess NYC taxi data"""

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] =  df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60 )

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def create_X(df: pd.DataFrame, dv: DictVectorizer | None = None):
    """Create feature matrix from dataframe using DictVectorizer"""

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

# def setup_mlflow():
#     """Initialize MLflow tracking URI and experiment"""
#     mlflow.set_tracking_uri(mlflow_tracking_uri)
#     mlflow.set_experiment(experiment_name)

def load_training_data(year: int, month: int) -> dict:
    """Load and preprocess training and validation data"""

    df_train = read_dataframe(year=year, month=month)

    #calculate validation month
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    # target = 'duration'
    # y_train = df_train[target].values
    # y_val = df_val[target].values

    return {
        'X_train': X_train,
        'y_train': df_train['duration'].values,
        'X_val': X_val,
        'y_val': df_val['duration'].values,
        'dv': dv,
        'train_size': len(df_train),
        'val_size': len(df_val)
    }

#Training and MLflow Logging
def train_model(X_train, y_train, X_val, y_val, dv, year: int, month: int) -> str:
    """Train XGBoost model and log parameters, metrics, and artifacts to MLflow"""

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        #Log training metadata to MLflow
        mlflow.set_tags({
            "training_year": year,
            "training_month": month,
            "model": "xgboost"
        })

        best_params = {
            'learning_rate': 0.095,
            'max_depth': 30,
            'min_child_weight': 1.060,
            'objective': 'reg:squarederror',
            'reg_alpha': 0.018,
            'reg_lambda': 0.011,
            'seed': 42
        }

        mlflow.log_params(best_params)
        
        mlflow.log_params({
            'train_year': year,
            'train_month': month,
            'num_boost_round': 30
        })

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        #Train the model with early stopping
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=20
        )

        #Metrics
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('rmse', rmse)

        #Log model to MLflow
        mlflow.xgboost.log_model(
            booster,
            artifact_path='model'
        )

        #Artifacts logging
        with tempfile.TemporaryDirectory() as tmp_dir:
            dv_path = Path(tmp_dir) / 'dict_vectorizer.pkl'

            with open(dv_path, 'wb') as f_out:
                pickle.dump(dv, f_out)

            mlflow.log_artifact(dv_path, artifact_path='preprocessor')

        return run.info.run_id
        # preprocessor_path = 'models/preprocessor.b'
        # with open(preprocessor_path, 'wb') as f_out:
        #     pickle.dump(dv, f_out)
        # mlflow.log_artifact(preprocessor_path, artifact_path='preprocessor')

        #Log Model artifact to MLflow
        # mlflow.xgboost.log_model(
        #     local_dir = models_folder, 
        #     artifact_path='models_mlflow')

        # return run.info.run_id

# def run_training_pipeline(year: int, month: int) -> str:
#     """Main training pipeline function to load data, train model, and return MLflow run ID"""

#     print(f"Starting training pipeline for {year}-{month:02d}")

#     #Load data
#     data = load_training_data(year, month)
#     print(f"Loaded {data['train_size']} training and {data['val_size']} validation records")

#     #Train model
#     run_id = train_model(
#         X_train=data['X_train'],
#         y_train=data['y_train'],
#         X_val=data['X_val'],
#         y_val=data['y_val'],
#         dv=data['dv'],
#         year=year,
#         month=month
#     )

#     print(f"Completed training pipeline for {year}-{month:02d} with MLflow run ID: {run_id}")
#     return run_id