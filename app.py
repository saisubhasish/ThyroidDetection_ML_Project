import sys
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from thyroid.predictor import ModelResolver
from thyroid.logger import logging
from thyroid.utils import load_object
from thyroid.exception import ThyroidException

app = Flask(__name__)


logging.info("Creating model resolver object")
model_resolver = ModelResolver(model_registry="saved_models")   # Location where models are saved


# Load the model
target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
model = load_object(file_path=model_resolver.get_latest_model_path())

@app.route('/')
def home():
    try:
        return render_template('home.html')
    
    except Exception as e:
        raise ThyroidException(error_message=e, error_detail=sys)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = [float(x) for x in request.form.values()]
        final_data = np.array(data).reshape(1,-1)
        logging.info(f"The input for the real time prediction: {final_data}")
        prediction = model.predict(final_data)
        cat_prediction = target_encoder.inverse_transform(prediction)
        print(cat_prediction)
        logging.info(f"The decoded output for the real time prediction: {cat_prediction}")
        
        return render_template('home.html', output_text="The prediction of Disease is: {}.".format(cat_prediction))
    
    except Exception as e:
        raise ThyroidException(error_message=e, error_detail=sys)


if __name__ == '__main__':
    app.run(debug=True)
