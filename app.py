import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import re
import pandas as pd
import matplotlib.dates as mdates
import pickle

app = Flask(__name__)
CORS(app)

def preprocess_text(text):
    try:
        # Remove URLs, HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', text)
        url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
        text = re.sub(url_pattern, '', text)

        return text
    except Exception as e:
        app.logger.error(f"Error occurred while preprocessing text: {e}")
    return text

def load_model(model_name, model_version):
    try:
        mlflow.set_tracking_uri("http://ec2-54-249-36-142.ap-northeast-1.compute.amazonaws.com:5000/")
        client = mlflow.tracking.MlflowClient()

        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.transformers.load_model(model_uri)
        return model
    except Exception as e:
        app.logger.error(f"Error loading model {model_name}: {e}")
        return None


model = load_model('Threads_Sentiment_Analysis_Model', '3')

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment
        preprocessed_comments = [preprocess_text(comment) for comment in comments]
        
        # Make predictions
        predictions = model(preprocessed_comments)
        print(predictions)
        response = []
        for original_comment, prediction in zip(comments, predictions):
            response.append({
                "comment": original_comment,
                "sentiment": str(prediction['label'])
            })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

