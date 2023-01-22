import sys
from thyroid.exception import ThyroidException
from thyroid.pipeline.training_pipeline import start_training_pipeline


if __name__ == "__main__":
     try:
         start_training_pipeline()

     except Exception as e:
          raise ThyroidException(error_message=e, error_detail=sys)