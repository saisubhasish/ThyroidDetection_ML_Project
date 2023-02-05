import os,sys 

from typing import Optional
from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from thyroid import utils
from thyroid.logger import logging
from thyroid.exception import ThyroidException
from thyroid.entity import artifact_entity,config_entity


class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise ThyroidException(e, sys)


    def fine_tune(self,x,y):
        """
        Hyper parameter tuning using GridSearchCV
        This function accepts x and y 
        -------------------------------------------
        Returns best paramenters for XGBClassifier
        """
        try:
            # Defining parameters
            parameters = {'max_depth': range (2, 10, 1),
                            'eta': [0.1, 0.2, 0.3],
                            'n_estimators': [100, 200, 300],
                            'learning_rate': [0.01, 0.03, 0.05]} 

            grid_search = GridSearchCV(estimator= XGBClassifier(), param_grid=parameters, n_jobs=-1, verbose=3, cv=3)
            
            grid_search.fit(x, y)
            BestParams = grid_search.best_params_
            
            return BestParams

        except Exception as e:
            raise ThyroidException(e, sys)

    def train_model(self,x,y):
        """
        Model training
        """
        try:
            xgb_clf =  XGBClassifier(eta= 0.1, learning_rate= 0.05, max_depth= 4, n_estimators= 300)
            xgb_clf.fit(x,y)
            return xgb_clf

        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        """
        Preparing dataset
        """
        try:
            logging.info("Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]

            logging.info('Hyperparameter tuning using GridSearchCV')
            Best_Params = self.fine_tune(x=x_train,y=y_train)
            print(f"The best parameters for XGBoostClassifier are : {Best_Params}")
            logging.info(f"The best parameters for XGBoostClassifier are : {Best_Params}")

            logging.info("Train the model")
            model = self.train_model(x=x_train,y=y_train)

            # Prediction and accuracy using training data
            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score= f1_score(y_true=y_train, y_pred=yhat_train)

            # Prediction and acuracy using test data
            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score= f1_score(y_true=y_test, y_pred=yhat_test)
            
            logging.info(f"train score:{f1_train_score} and tests score {f1_test_score}")
            logging.info("Checking if our model is a good model or not")
            if f1_test_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {f1_test_score}")

            logging.info("Checking if our model is overfiiting or not")
            diff = abs(f1_train_score-f1_test_score)   # Checking the difference by removing -ve

            # Check for overfitting or underfiiting on threshold
            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            # Saving trained model if it passes using utils
            logging.info("Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare artifact
            logging.info("Prepare the artifact")
            model_trainer_artifact  = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise ThyroidException(e, sys)

