
import pickle
import pandas as pd
import argparse

#CLI Args
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
args = parser.parse_args()

year = args.year
month = args.month


#Load model
with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


#Load current month
filename = f"./data/yellow_tripdata_{year}-{month:02d}.parquet"
df = read_data(filename)

# df = read_data('./data/yellow_tripdata_2023-04.parquet')
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(y_pred.mean())


# year = 2023
# month = 3

# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# df_result = pd.DataFrame({
#     'ride_id': df['ride_id'],
#     'predicted_duration': y_pred
# })

# output_file = './data/yellow_tripdata_2023-03_predictions.parquet'

# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     index=False,
#     compression=None
# )

# os.path.getsize(output_file) / 1024 / 1024

