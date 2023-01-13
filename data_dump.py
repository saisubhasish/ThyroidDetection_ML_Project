import json
import pymongo
import pandas as pd
from thyroid.config import mongo_client

DATA_FILE_PATH="/config/workspace/hypothyroid.csv"
DATABASE_NAME="HealthCare"
COLLECTION_NAME="Thyroid"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    # Each record will represent one row
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[1])
    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
