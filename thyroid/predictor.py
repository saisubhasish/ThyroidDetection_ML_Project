import os, sys
from thyroid.entity.config_entity import TARGET_ENCODER_OBJECT_FILE_NAME, MODEL_FILE_NAME
from glob import glob
from typing import Optional
from thyroid.exception import ThyroidException


class ModelResolver:
    """
    This class is helping us to get the location of required updated files (where to save the model 
    and from where to load the model) for prediction pipeline
    """
    def __init__(self,model_registry:str = "saved_models",
                target_encoder_dir_name = "target_encoder",
                model_dir_name = "model"):

        self.model_registry=model_registry
        os.makedirs(self.model_registry,exist_ok=True)
        self.target_encoder_dir_name=target_encoder_dir_name
        self.model_dir_name=model_dir_name


    def get_latest_dir_path(self)->Optional[str]:
        """
        This function returns None if there is no saved_models present
        Otherwise returns the path of the latest saved_models directory
        """
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names)==0:
                return None
            dir_names = list(map(int,dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registry,f"{latest_dir_name}")
        except Exception as e:
            raise ThyroidException(e, sys)

    def get_latest_model_path(self):
        """
        This function raise Exception if there is no model present in saved models dir
        Otherwise returns the path of the latest model present in saved_models directory
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is not available")
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)


    def get_latest_target_encoder_path(self):
        """
        This function raise Exception if there is no Target Encoder present in saved models dir
        Otherwise returns the path of the latest Target Encoder present in saved_models directory
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Target encoder is not available")
            return os.path.join(latest_dir,self.target_encoder_dir_name,TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)


    def get_latest_save_dir_path(self)->str:
        """
        This function returns 0 if there is no saved_models dir present
        Otherwise return by adding a number to pre-exist directory 
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir==None:  # If there is no pre-exist directory then create a directory as 0
                return os.path.join(self.model_registry,f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry,f"{latest_dir_num+1}") # Otherwise creating a directory with a number addition
        except Exception as e:
            raise ThyroidException(e, sys)

    def get_latest_save_model_path(self):
        """
        This function extracts the latest saved_models directory and returns the path to save the latest model
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,MODEL_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)


    def get_latest_save_target_encoder_path(self):
        """
        This function extracts the latest saved_models directory and returns the path to save the latest Target Encoder
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.target_encoder_dir_name,TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise ThyroidException(e, sys)


    