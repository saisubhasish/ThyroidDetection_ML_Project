import pymongo
import pandas as pd
import json 
from dataclasses import dataclass
import os

@dataclass
class EnvironmentVariable:
    """
    In this class we are accessing the variables declared in .env file
    """
    mongo_db_url:str = os.getenv("MONGO_DB_URL")
    aws_access_key_id:str = os.getenv("AWS_ACCESS_KEY_ID")
    aws_access_secret_key:str = os.getenv("AWS_SECRET_ACCESS_KEY")



env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = "binaryClass"

