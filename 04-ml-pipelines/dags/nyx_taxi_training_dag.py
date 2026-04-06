from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable


import sys
sys.path.insert(0, '/opt/airflow/src')

#Default arguments for the DAG
default_args = {
    'owner': 'avad-mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email': ['avadhoothublikar109@gmail.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def task_load_data(**context):
    """Load and preprocess training and validation data"""
    from training import load_training_data

    #Get parameters from DAG run config or use defaults
    dag_run = context.get('dag_run')
    params = dag_run.conf if dag_run and dag_run.conf else {}

    year = params.get('year', 2023)
    month = params.get('month', 1)

    print(f"Loading data for year: {year}, month: {month:02d}")

    data = load_training_data(year, month)

    #store metadata (not the actual data it is too large for xcom)
    context['ti'].xcom_push(key='train_year', value=year)
    context['ti'].xcom_push(key='train_month', value=month)
    context['ti'].xcom_push(key='train_size', value=data['train_size'])
    context['ti'].xcom_push(key='val_size', value=data['val_size'])

    print(f"Loaded {data['train_size']} training and {data['val_size']} validation records")


def task_train_model(**context):
    """Train XGBoost model and log results to MLflow"""
    from training import load_training_data, train_model

    ti = context['ti']

    year = ti.xcom_pull(key='train_year')
    month = ti.xcom_pull(key='train_month')

    #Reload data (do not use xcom to pass large data)
    data = load_training_data(year, month)

    run_id = train_model(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        dv=data['dv'],
        year=year,
        month=month
    )

    #store run_id in xcom for downstream tasks
    ti.xcom_push(key='mlflow_run_id', value=run_id)
    print(f"Model training completed with MLflow run ID: {run_id}")


def task_validate_model(**context):
    """Validate the trained model meets quality thresholds"""
    import mlflow

    ti = context['ti']
    run_id = ti.xcom_pull(key='mlflow_run_id')

    #connect to mlflow and get the rmse metric
    mlflow.set_tracking_uri('http://mlflow:5000')
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    rmse = run.data.metrics.get('rmse')

    print(f"Model RMSE: {rmse}")

    #Define a threshold for model performance
    rmse_threshold = 10.0

    if rmse > rmse_threshold:
        raise ValueError(f"Model RMSE {rmse} exceeds threshold of {rmse_threshold}")
    
    ti.xcom_push(key='model_rmse', value=rmse)
    ti.xcom_push(key='model_validated', value=True)

    print(f"Model validated successfully with RMSE: {rmse}")

def task_register_model(**context):
    """Register the validated model in MLflow Model Registry"""
    import mlflow

    ti = context['ti']
    run_id = ti.xcom_pull(key='mlflow_run_id')
    year = ti.xcom_pull(key='train_year')
    month = ti.xcom_pull(key='train_month')

    mlflow.set_tracking_uri('http://mlflow:5000')
    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    model_name = 'nyx_taxi_duration_model'


    #Register the model
    registered_model = mlflow.register_model(model_uri, model_name)
    version = registered_model.version

    #Add description
    client.update_model_version(
        name=model_name,
        version=version,
        description=f"Model trained on {year}-{month:02d} data with RMSE {ti.xcom_pull(key='model_rmse')}"
    )

    ti.xcom_push(key='model_version', value=version)
    print(f"Model registered successfully as {model_name} version {version}")


def task_notify_completion(**context):
    """Send notification of pipeline completion"""
    ti = context['ti']

    run_id = ti.xcom_pull(key='mlflow_run_id')
    year = ti.xcom_pull(key='train_year')
    month = ti.xcom_pull(key='train_month')
    rmse = ti.xcom_pull(key='model_rmse')
    model_version = ti.xcom_pull(key='model_version')

    message = f"""
    NYC Taxi Model Training Completed
    ---------------------------------------------
    Training Data: {year}-{month:02d}
    MLflow Run ID: {run_id}
    Model RMSE: {rmse}
    Registered Model Version: {model_version}
    View in MLflow: http://localhost:5000/#/experiments/1/runs/{run_id}
    """

    print(message)

with DAG(
    dag_id='nyx_taxi_training_dag',
    default_args=default_args,
    description='Train XGBoost model for NYC taxi duration prediction',
    schedule_interval='0 0 1 * *',  #Run on the first day of every month
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'training', 'xgboost','nyc-taxi'],
    params={
        'year': 2023,
        'month': 1
    }
) as dag:
    
    start = EmptyOperator(task_id='start')

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=task_load_data,
        provide_context=True,
    )


    train_model = PythonOperator(
        task_id='train_model',
        python_callable=task_train_model,
        provide_context=True,
    )

    validate_model = PythonOperator(
        task_id = 'validate_model',
        python_callable=task_validate_model,
        provide_context=True,
    )

    register_model = PythonOperator(
        task_id='register_model',
        python_callable=task_register_model,
        provide_context=True,
    )

    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=task_notify_completion,
        provide_context=True,
    )

    end = EmptyOperator(task_id='end')

    #Define pipeline flow
    start >> load_data >> train_model >> validate_model >> register_model >> notify >> end
    