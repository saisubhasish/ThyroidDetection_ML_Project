import os,sys
import pandas as pd
import numpy as np
from datetime import datetime

from thyroid import utils
from thyroid.logger import logging
from thyroid.utils import load_object
from thyroid.config import TARGET_COLUMN
from thyroid.entity import config_entity
from thyroid.predictor import ModelResolver
from thyroid.exception import ThyroidException
from thyroid.components.data_validation import DataValidation
from thyroid.entity.config_entity import DataValidationConfig


PREDICTION_DIR= "prediction"
VALIDATION_DIR= "validation_report"

validation_error=dict()


base_file_path = os.path.join("hypothyroid.csv")


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok=True)
        report_file_dir = os.path.join(PREDICTION_DIR, VALIDATION_DIR)
        os.makedirs(report_file_dir, exist_ok=True)

        logging.info("Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")   # Location where models are saved
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        base_df= pd.read_csv(base_file_path)
        logging.info("Replace '?' value to nan in base and input df")
        df.replace({"?":np.NAN},inplace=True)
        base_df.replace({"?":np.NAN},inplace=True)
        
        # Validation
        logging.info("Validating input file")
        try:
            logging.info("Dropping missing value columns from current df")
            df = model_resolver.drop_missing_values_columns(df=df,report_key_name="missing_values_within_input_df")
            logging.info("Dropping missing value columns from base df")
            base_df = model_resolver.drop_missing_values_columns(df=base_df,report_key_name="missing_values_within_base_dataset")

            logging.info("Checking required columns in current df")
            current_column_status = model_resolver.is_required_columns_exists(base_df=base_df, current_df=df, report_key_name="missing_columns_within_input_dataset")

            logging.info("Checking data drift current df")
            if current_column_status:
                validation_error = model_resolver.data_drift(base_df=base_df, current_df=df, report_key_name="data_drift_within_input_dataset")

            # Creating directory to save prediction file
            report_file_path = os.path.join(report_file_dir,"report.yaml")

            utils.write_yaml_file(file_path=report_file_path, data=validation_error)

        except Exception as e:
            raise ThyroidException(e, sys)

        logging.info("Data Transformation")
        try:
            df = model_resolver.feature_encoding(df=df)
            exclude_columns = [TARGET_COLUMN]
            utils.convert_columns_float(df=df, exclude_columns=exclude_columns)
            df = model_resolver.handling_null_value_and_outliers(df=df)

        except Exception as e:
            raise ThyroidException(e, sys)

        # Loading knn_imputer
        logging.info("Loading knn imputer to get dataset")
        knn_imputer = load_object(file_path=model_resolver.get_latest_knn_imputer_path())
        
        # Getting input features
        input_feature_names =  list(knn_imputer.feature_names_in_)
        # data frame
        input_arr = knn_imputer.transform(df[input_feature_names])

        # Prediction    
        logging.info("Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        # Target decoding   
        logging.info("Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

        cat_prediction = target_encoder.inverse_transform(prediction)

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction

        logging.info('Creating prediction file with time stamp')
        # Creating file name for predition with time stamp by replacing .csv
        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        # Creating directory to save prediction file
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        # Saving the df to directory
        df.to_csv(prediction_file_path,index=False,header=True)

        return prediction_file_path

    except Exception as e:
        raise ThyroidException(e, sys)
