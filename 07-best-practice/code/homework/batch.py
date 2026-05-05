#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pickle
import pandas as pd

def get_input_path(year, month):
    default_input_pattern = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = (
        "s3://nyc-duration-prediction-alexey/"
        "taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    )
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)

def get_s3_storage_options():
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if not s3_endpoint_url:
        return None

    return {
        "client_kwargs": {
            "endpoint_url": s3_endpoint_url
        }
    }

def prepare_data(df, categorical):
    df = df.copy()

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def read_data(filename, categorical):
    storage_options = get_s3_storage_options()

    if filename.startswith("s3://") and storage_options is not None:
        df = pd.read_parquet(filename, storage_options=storage_options)
    else:
        df = pd.read_parquet(filename)

    return prepare_data(df, categorical)

def main(year, month):
    input_file = get_input_path(year, month)    
    output_file = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    storage_options = get_s3_storage_options()
    if output_file.startswith("s3://") and storage_options is not None:
        df_result.to_parquet(
            output_file,
            engine='pyarrow', 
            index=False, 
            storage_options=storage_options
        )
    else:
        df_result.to_parquet(output_file, engine='pyarrow', index=False)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)