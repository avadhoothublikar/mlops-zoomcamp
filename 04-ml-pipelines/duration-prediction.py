#!/usr/bin/env python

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error


import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    # df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    # df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] =  df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60 )

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):

    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv



def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.095,
            'max_depth': 30,
            'min_child_weight': 1.060,
            'objective': 'reg:linear',
            'reg_alpha': 0.018,
            'reg_lambda': 0.011,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=20
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('rmse', rmse)


        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('models/preprocessor.b', artifact_path='preprocessor')

        mlflow.xgboost.log_model(booster, artifact_path='models_mlflow')  

        #return run_id from mlfow
        return run.info.run_id

def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    x_train, dv = create_X(df_train)
    x_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id =train_model(x_train, y_train, x_val, y_val, dv)
    print(f"MLflow run ID: {run_id}")
    return run_id

if __name__ == '__main__':
    #use argparse to get the year and month
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict the duration of a taxi trip')
    parser.add_argument('--year', type=int, required=True, help='The year of the data to use for training')
    parser.add_argument('--month', type=int, required=True, help='The month of the data to use for training') 
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    #save run_id to a file
    with open('run_id.txt', 'w') as f_out:
        f_out.write(run_id)





