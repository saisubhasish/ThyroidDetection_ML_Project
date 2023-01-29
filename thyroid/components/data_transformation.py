import os,sys 
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from typing import Optional
from imblearn.combine import SMOTETomek    # To generate some data for minority class 

from thyroid import utils
from thyroid.entity import artifact_entity,config_entity
from thyroid.exception import ThyroidException
from thyroid.logger import logging
from thyroid.config import TARGET_COLUMN



class DataTransformation:

    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_validation_artifact:artifact_entity.DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
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
            logging.info("Replacing 'f' to 0 and 't' to 1 in dataframe")
            df = df.replace({'f':0, 't':1})

            logging.info("Replacing 'F' to 0 and 'M' to 1 'sex' column")
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

    @classmethod
    def get_knn_imputer_object(cls)->Pipeline:     # Attributes of this class will be same across all the object 
        try:
            knn_imputer = KNNImputer(n_neighbors=7)
            knn_pipeline = Pipeline(steps=[
                    ('imputer', knn_imputer)    # To populate data for missing rows
                ])
            return knn_pipeline

        except Exception as e:
            raise ThyroidException(e, sys)
    

    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            # Reading training and testing file
            logging.info("Reading training and testing file")
            train_df = pd.read_csv(self.data_validation_artifact.train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.test_file_path)
            
            # Converting the categorical data to numerical
            logging.info("Converting the categorical data to numerical")
            train_df = self.feature_encoding(df=train_df)
            test_df = self.feature_encoding(df=test_df)            
            
            # Converting columns datatypes to float type
            logging.info("Converting columns datatypes to float type")
            exclude_columns = [TARGET_COLUMN]
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            # Handling outlier and null value in 'age' and 'sex' column
            logging.info("Handling outlier and null value in 'age' and 'sex' column")
            train_df = self.handling_null_value_and_outliers(df=train_df)
            test_df = self.handling_null_value_and_outliers(df=test_df)
            
            # Selecting input feature for train and test dataframe
            logging.info("Selecting input feature for train and test dataframe")
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            # Selecting target feature for train and test dataframe
            logging.info("Selecting target feature for train and test dataframe")
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Encoding the target feature values
            logging.info("Encoding the target feature values")
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)

            # Transformation on target columns
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)   
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)
            logging.info(f"Target feature label encoded values: {target_feature_test_arr}")

            # Imputing null values with KNNImputer
            imputation_pipeline = DataTransformation.get_knn_imputer_object()
            imputation_pipeline.fit_transform(input_feature_train_df)
            logging.info(input_feature_train_df.columns)
            
            # Imputing null values
            input_feature_train_arr = imputation_pipeline.transform(input_feature_train_df)  
            features_names = list(imputation_pipeline.feature_names_in_)                      #####    To handle the features in test set
            input_feature_test_df = input_feature_test_df[features_names]
            input_feature_test_arr = imputation_pipeline.transform(input_feature_test_df)        ##### As 'T3' is present in test_df
            
            # Handling imbalanced data by resampling
            smt = SMOTETomek(random_state=42,sampling_strategy='minority')
            logging.info(f"Before resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input: {input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr, target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input: {input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            # Target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]    # concatenated transpose array
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            # Save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            # Saving object
            utils.save_object(file_path=self.data_transformation_config.knn_imputer_object_path, obj=imputation_pipeline)
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=label_encoder)

            # Preparing Artifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                knn_imputer_object_path=self.data_transformation_config.knn_imputer_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                target_encoder_path = self.data_transformation_config.target_encoder_path)

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise ThyroidException(e, sys)
