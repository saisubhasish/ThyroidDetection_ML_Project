# This code is from airflow
from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG           # To schedule anything in airflow you need you define a DAG for it
from airflow.operators.python import PythonOperator


with DAG(                         # Directed Acyclic Graph
    'thyroid_training',            # Name for DAG
    default_args={'retries': 2},  # No of retries for pipeline, if pipeline 1st time fails it will try for second, after that it is a failure
    # [END default_args]
    description='Thyroid Disease Detection',
    schedule_interval="@weekly",  # Pipeline will run weekly basis
    start_date=pendulum.datetime(2023, 2, 2, tz="UTC"),   # Start date for pipeline
    catchup=False,              # Flag relation to previous run
    tags=['example'],           # list of string
) as dag:

    
    def training(**kwargs):                                                      # Defining function for training pipeline
        from thyroid.pipeline.training_pipeline import start_training_pipeline    # Starting training pipeline from thyroid
        start_training_pipeline()
    
    def sync_artifact_to_s3_bucket(**kwargs):                                    # Defining function to Store artifacts and models to S3 bucket
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifacts")      # Storing artifacts to in buctet
        os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models") # Saving models to bucket

    training_pipeline  = PythonOperator(               # We are using python operator to call training function
            task_id="train_pipeline",                  # so that when we will run this code from airflow pipeline 
            python_callable=training                   # it will trigger training pipeline

    )

    sync_data_to_s3 = PythonOperator(                  # Calling using python operator
            task_id="sync_data_to_s3",
            python_callable=sync_artifact_to_s3_bucket

    )

    training_pipeline >> sync_data_to_s3                # The flow of execution