FROM apache/airflow:2.7.0-python3.10

USER airflow

# Install compatible versions together
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    pyarrow==12.0.1 \
    scikit-learn==1.3.0 \
    xgboost==1.7.6 \
    mlflow==2.5.0