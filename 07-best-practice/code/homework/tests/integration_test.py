import os
from datetime import datetime
import pandas as pd
import subprocess
from pathlib import Path
import sys

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_integration():
    options = {
        "client_kwargs": {
            "endpoint_url": "http://localhost:4566"
        }
    }


    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime"
    ]

    df_input = pd.DataFrame(data, columns=columns)

    input_file = "s3://nyc-duration/in/2023-01.parquet"
    output_file = "s3://nyc-duration/out/2023-01.parquet"

    df_input.to_parquet(
        input_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options=options
    )

    os.environ['INPUT_FILE_PATTERN'] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    os.environ['OUTPUT_FILE_PATTERN'] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
    os.environ['S3_ENDPOINT_URL'] = "http://localhost:4566"
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    project_dir = Path(__file__).resolve().parents[1]
    batch_path = project_dir / 'batch.py'

    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, str(batch_path), '2023', '1'],
        cwd=project_dir,
        check=False,
        env=env,
        text=True,
        capture_output=True
    )

    assert result.returncode == 0, (
        "batch.py failed\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    df_result = pd.read_parquet(
        output_file,
        storage_options=options
    )

    total = round(df_result['predicted_duration'].sum(), 2)
    assert total == 36.28