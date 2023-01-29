import os, sys
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from thyroid import utils
from thyroid.logger import logging
from thyroid.utils import load_object
from thyroid.config import TARGET_COLUMN
from thyroid.predictor import ModelResolver
from thyroid.exception import ThyroidException
from thyroid.entity import config_entity, artifact_entity
from thyroid.components.data_transformation import DataTransformation
 

class ModelEvaluation:

    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            logging.info("___________________________________________________________________________________________________________")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.data_transformation= DataTransformation(data_transformation_config=config_entity.DataTransformationConfig, data_validation_artifact=artifact_entity.DataValidationArtifact)
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise ThyroidException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            # If saved model folder has model then we will compare which model is best
            # Trained model from artifact folder or the model from saved model folder
            logging.info("___________________________________________________________________________________________________________")
            logging.info("If saved model folder has model then we will compare which model is best, "
            "Trained model from artifact folder or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:                                 # If there is no saved_models then we will accept the currnt model
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)                                           
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of model and target encoder
            logging.info("Finding location of knn_imputer, model and target encoder")
            knn_imputer_path = self.model_resolver.get_latest_knn_imputer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            # Loading objects
            logging.info("Previous trained objects of knn_imputer, model and target encoder")
            # Previous trained  objects
            knn_imputer = load_object(file_path=knn_imputer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)
            
            logging.info("Currently trained model objects")
            # Currently trained model objects
            current_knn_imputer = load_object(file_path=self.data_transformation_artifact.knn_imputer_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            
            # Reading test file
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            # output label
            target_df = test_df[TARGET_COLUMN]
            y_true =target_encoder.transform(target_df)
            
            # Accuracy using previous trained model
            exclude_columns = [TARGET_COLUMN]
            input_feature_name = list(knn_imputer.feature_names_in_)
            input_feature_test_df= test_df[input_feature_name]
            input_feature_test_df= self.data_transformation.feature_encoding(df=input_feature_test_df)
            input_feature_test_df= utils.convert_columns_float(df=input_feature_test_df, exclude_columns=exclude_columns)
            input_feature_test_df= self.data_transformation.handling_null_value_and_outliers(df=input_feature_test_df)

            input_arr= knn_imputer.transform(input_feature_test_df)
            y_pred = model.predict(input_arr)

            # Label decoding with 5 values to get actual string
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")

            # Accuracy using current trained model
            exclude_columns = [TARGET_COLUMN]
            input_feature_name = list(current_knn_imputer.feature_names_in_)
            input_feature_test_df= test_df[input_feature_name]
            input_feature_test_df= self.data_transformation.feature_encoding(df=input_feature_test_df)
            input_feature_test_df= utils.convert_columns_float(df=input_feature_test_df, exclude_columns=exclude_columns)
            input_feature_test_df= self.data_transformation.handling_null_value_and_outliers(df=input_feature_test_df)

            input_arr= current_knn_imputer.transform(input_feature_test_df)
            y_pred= current_model.predict(input_arr)
            y_true= current_target_encoder.transform(target_df)
            # Label decoding with 5 values to get actual string 
            print(f"Prediction using trained model: {current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model: {current_model_score}")
            if current_model_score<=previous_model_score:
                logging.info("Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            
            # Improved accuracy
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
            
        except Exception as e:
            raise ThyroidException(e,sys)
