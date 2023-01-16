import os, sys
from thyroid.logger import logging
from thyroid.exception import ThyroidException
from thyroid.utils import get_collection_as_dataframe
from thyroid.entity import config_entity, artifact_entity
from thyroid.components.data_ingestion import DataIngestion
from thyroid.components.data_validation import DataValidation
from thyroid.entity.config_entity import DataIngestionConfig
from thyroid.entity.config_entity import DataValidationConfig
from thyroid.components.feature_engineering import FeatureEngineering
from thyroid.entity.config_entity import FeatureEngineeringConfig




if __name__ == "__main__":
     try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()

        #data ingestion         
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


        #data validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config,
                        data_ingestion_artifact=data_ingestion_artifact)

        data_validation_artifact = data_validation.initiate_data_validation()

        #feature engineering
        feature_engineering_config = config_entity.FeatureEngineeringConfig(training_pipeline_config=training_pipeline_config)
        feature_engineering = FeatureEngineering(feature_engineering_config=feature_engineering_config, 
        data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = feature_engineering.initiate_feature_engineering()

        

     except Exception as e:
          raise ThyroidException(error_message=e, error_detail=sys)