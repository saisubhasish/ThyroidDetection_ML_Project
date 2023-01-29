import os, sys
from thyroid.logger import logging
from thyroid.exception import ThyroidException
from thyroid.predictor import ModelResolver
from thyroid.entity.config_entity import ModelPusherConfig
from thyroid.utils import save_object, load_object
from thyroid.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelPusherArtifact

class ModelPusher:

    def __init__(self,model_pusher_config:ModelPusherConfig,
        data_transformation_artifact:DataTransformationArtifact,
        model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config=model_pusher_config
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            # Load object 
            logging.info("Loading knn_imputer, model and target encoder")
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            knn_imputer = load_object(file_path=self.data_transformation_artifact.knn_imputer_object_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # Model pusher dir
            logging.info("Saving model into model pusher directory")
            save_object(file_path= self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.knn_imputer_object_path, obj=knn_imputer)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)

            # Getting or fetching the directory location to save latest model in different directory in each run
            logging.info("Saving model in saved model dir")
            model_path = self.model_resolver.get_latest_save_model_path()
            knn_imputer_path = self.model_resolver.get_latest_save_knn_imputer_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            # Saved model dir outside artifact to use in prediction pipeline
            logging.info('Saving model outside of artifact directory')
            save_object(file_path=model_path, obj=model)
            save_object(file_path=knn_imputer_path, obj=knn_imputer)
            save_object(file_path=target_encoder_path, obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir, 
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact : {model_pusher_artifact}")

            return model_pusher_artifact

        except Exception as e:
            raise ThyroidException(e, sys)
