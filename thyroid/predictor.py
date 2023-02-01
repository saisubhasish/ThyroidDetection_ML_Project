import os, sys
import pandas as pd
import numpy as np
from thyroid.entity.config_entity import TARGET_ENCODER_OBJECT_FILE_NAME, MODEL_FILE_NAME, KNN_IMPUTER_OBJECT_FILE_NAME
from typing import Optional
from thyroid.exception import ThyroidException

missing_threshold = 0.2
validation_error=dict()


class ModelResolver:
    """
    This class is helping us to get the location of required updated files (where to save the model 
    and from where to load the model) for prediction pipeline
    """
    def __init__(self,model_registry:str = "saved_models",
                target_encoder_dir_name = "target_encoder",
                knn_imputer_dir_name = "knn_imputer",
                model_dir_name = "model"):

        self.model_registry=model_registry
        os.makedirs(self.model_registry,exist_ok=True)
        self.target_encoder_dir_name=target_encoder_dir_name
        self.model_dir_name=model_dir_name
        self.knn_imputer_dir_name= knn_imputer_dir_name


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

    def get_latest_knn_imputer_path(self):
        """
        This function raise Exception if there is no Transformer present in saved models dir
        Otherwise returns the path of the latest Transformer present in saved_models directory
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"KNN Imputer is not available")
            return os.path.join(latest_dir,self.knn_imputer_dir_name,KNN_IMPUTER_OBJECT_FILE_NAME)
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

    def get_latest_save_knn_imputer_path(self):
        """
        This function extracts the latest saved_models directory and returns the path to save the latest KNN Imputer
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.knn_imputer_dir_name,KNN_IMPUTER_OBJECT_FILE_NAME)

        except Exception as e:
            raise ThyroidException(e, sys)

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold
        df : Accepts a pandas dataframe
        =========================================================================================
        returns Pandas Dataframe if atleast a single column is available after missing columns drop else None
        """
        try:
            
            threshold = missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            # Selecting column name which contains null
            drop_column_names = null_report[null_report>threshold].index

            validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            # Return None if no columns left
            if len(df.columns)==0:
                return None

            return df

        except Exception as e:
            raise ThyroidException(e, sys)


    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str)->bool:
        """
        This function checks if required columns exists or not by comparing current df with base df and returns
        output as True and False
        """
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:    
                if base_column not in current_columns:
                    missing_columns.append(base_column)

            # Return False if there are missing columns in current df other wise True
            if len(missing_columns)>0:
                validation_error[report_key_name]=missing_columns
                return False   

            return True
            
        except Exception as e:
            raise ThyroidException(e, sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            drift_report=dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data,current_data = base_df[base_column],current_df[base_column]
                # Null hypothesis : Both column data has same distribution
                
                if base_df[base_column].dtype == current_df[base_column].dtype:
                    drift_report[base_column] = {"Same data type": True}
                else:
                    drift_report[base_column] = {"Same data type": False}

                if len(base_df[base_column].value_counts()) == len(current_df[base_column].value_counts()):
                    drift_report[base_column] = {"Column has equal number of classes": True}
                else:
                    drift_report[base_column] = {"Column has equal number of classes": False} 

            validation_error[report_key_name]=drift_report
            
            return validation_error
            
        except Exception as e:
            raise ThyroidException(e, sys)

    def feature_encoding(self,df:pd.DataFrame)->Optional[pd.DataFrame]:
        """
        This function will replace the categorical data of each column to numerical (Array type)

        df : Accepts a pandas dataframe
        =========================================================================================
        returns Pandas Dataframe after converting to numerical value
        """
        try:
            df = df.replace({'f':0, 't':1})

            df['sex'] = df['sex'].replace({'F':0, 'M':1})
            return df

        except Exception as e:
            raise ThyroidException(e, sys)

    def handling_null_value_and_outliers(self,df:pd.DataFrame)->Optional[pd.DataFrame]:
        """
        This function will fill median in 'age' to handle outlier and null and mode in 'sex' column for null value

        df : Accepts a pandas dataframe
        ==========================================================================================================
        returns Pandas Dataframe after filling the value
        """
        try:
            median = df.loc[df['age']<=94, 'age'].median()
            df.loc[df.age > 94, 'age'] = np.nan
            df['age'].fillna(median,inplace=True)

            df['sex'] = df['sex'].replace(np.nan, df['sex'].mode()[0])

            return df

        except Exception as e:
            raise ThyroidException(e, sys)






    