import logging
import os
import json
import mlflow

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path):
    """Load model information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
            logger.info(f"Model information loaded successfully from {file_path}")
            return model_info
    except FileNotFoundError:
        logger.error(f"Model info file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def register_model(model_name, model_info):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # register the model
        result = mlflow.register_model(model_uri, model_name)

        # To staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=result.name,
            version=result.version,
            stage="Staging"
        )
        logger.info(f"Model {model_name} version {result.version} registered and moved to Staging")

    except Exception as e:
        logger.error(f"Error registering model {model_name}: {e}")
        raise

def main():
    try:
        mlflow.set_tracking_uri("http://ec2-54-249-36-142.ap-northeast-1.compute.amazonaws.com:5000/") 
        model_info = load_model_info('experiment_info.json')
        register_model('Threads_Sentiment_Analysis_Model', model_info)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise

if __name__ == "__main__":
    main()