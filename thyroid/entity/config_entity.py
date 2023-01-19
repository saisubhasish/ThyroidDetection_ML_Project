import os, sys
from datetime import datetime
from thyroid.logger import logging
from thyroid.exception import ThyroidException

FILE_NAME = 'thyroid.csv'
TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'
IMPUTER_OBJECT_FILE_NAME = "knn_imputer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"


class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")

        except Exception as e:
            raise ThyroidException(e, sys)

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name="HealthCare"
            self.collection_name="Thyroid"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir , "data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2

        except Exception as e:
            raise ThyroidException(e, sys)

    def to_dict(self,)->dict:
        """
        To convert and return the output as dict : data_ingestion_config
        """ 
        try:
            return self.__dict__

        except Exception  as e:
            raise SensorException(e,sys) 

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "data_validation")
            self.report_file_path=os.path.join(self.data_validation_dir, "report.yaml")
            self.missing_threshold:float = 0.2
            self.base_file_path = os.path.join("hypothyroid.csv")

        except Exception as e:
            raise ThyroidException(e, sys)

class FeatureEngineeringConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.feature_engineering_dir = os.path.join(training_pipeline_config.artifact_dir , "feature_engineering")
            self.imputer_object_path = os.path.join(self.feature_engineering_dir,"imputer",IMPUTER_OBJECT_FILE_NAME)
            self.transformed_train_path =  os.path.join(self.feature_engineering_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
            self.transformed_test_path =os.path.join(self.feature_engineering_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))
            self.target_encoder_path = os.path.join(self.feature_engineering_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise ThyroidException(e, sys)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
            self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
            self.expected_score = 0.7
            self.overfitting_threshold = 0.1

        except Exception as e:
            raise ThyroidException(e, sys)

class ModelEvaluationConfig:...
class ModelPusherConfig:...