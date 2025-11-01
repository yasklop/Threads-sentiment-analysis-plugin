import logging
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml
import json
from model_training import compute_metrics, load_config, tokenize_data, TweetDataset, load_data
import mlflow
from mlflow.models import infer_signature
import numpy as np

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(model_path):
    """Load a trained model from the specified path."""
    try:
        current_dir = os.path.abspath(os.getcwd())
        checkpoint_path = os.path.join(current_dir, model_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=4, local_files_only=True)
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise

def evaluate_model(model, test_dataset):
    """Evaluate the model on the test dataset."""
    try:
        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics
        )
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"Model evaluation completed successfully: {eval_result}")
        return eval_result
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_model_info(id, model_path, file_path):
    """Save model information to a text file."""
     # Create a dictionary with the info you want to save
    try:
        model_info = {
            'run_id': id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    try:
        mlflow.set_tracking_uri("http://ec2-54-249-36-142.ap-northeast-1.compute.amazonaws.com:5000/") 

        EXPERIMENT_NAME = "BERT_Sentiment_Analysis"
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Windows
        import sys

        sys.stdout.reconfigure(encoding='utf-8')

        full_config = load_config('params.yaml')
        config = full_config['model_params']
        df = load_data('data/preprocessed/final_data.csv')

        X = df['Tweet'].to_list()
        y = df['Sentiment']

        MODEL_NAME = config['model_name']
        MAX_LENGTH = config['max_length']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config['split_ratio'], random_state=42)

        val_encodings = tokenize_data(MODEL_NAME, X_val, MAX_LENGTH)
        val_dataset = TweetDataset(val_encodings, y_val)

        model = load_model('results/current_model')
        

        with mlflow.start_run() as run:

            trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics
            )
            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            
            # Log model
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            input_example = pd.DataFrame({
                "text": ["I love this product!", "This is the worst service ever."]
            })

            analysis_pipeline = pipeline(
                "text-classification", 
                model=model, 
                tokenizer=tokenizer,
                return_all_scores=True
            )
            
            preds = analysis_pipeline(input_example['text'].tolist(), truncation=True)
            # convert to numpy array shape (n_samples, n_labels)
            probs = np.array([[d["score"] for d in item] for item in preds])

            col_names = [f"prob_label_{i}" for i in range(probs.shape[1])]
            output_example = pd.DataFrame(probs, columns=col_names)

            signature = infer_signature(input_example, output_example)
            mlflow.transformers.log_model(
                transformers_model=analysis_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            
            model_path = "model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # Add important tags
            mlflow.set_tag("model_type", "BERT")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "Threads Comments")
            
        logger.info("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()

