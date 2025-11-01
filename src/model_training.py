import logging
import os
import pandas as pd
import yaml
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_training_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_config(config_path):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

def tokenize_data(model_name, texts, max_length):
    """Tokenize the text data."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        texts = [str(text) for text in texts]  # Ensure all texts are strings
        tokens = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt' 
        )
        logger.info("Data tokenization completed successfully")
        return tokens
    except Exception as e:
        logger.error(f"Error during data tokenization: {e}")
        raise

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def build_model(config, X, y):
    """Build the model."""
    try:
        MODEL_NAME = config['model_name']
        MAX_LENGTH = config['max_length']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config['split_ratio'], random_state=42)

        train_encodings = tokenize_data(MODEL_NAME, X_train, MAX_LENGTH)
        val_encodings = tokenize_data(MODEL_NAME, X_val, MAX_LENGTH)

        train_dataset = TweetDataset(train_encodings, y_train)
        val_dataset = TweetDataset(val_encodings, y_val)     

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=config['num_epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            eval_strategy='epoch',
            save_strategy='epoch',
            report_to="mlflow",
            run_name="bert",
            learning_rate=float(config['learning_rate']),
            metric_for_best_model='accuracy',
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("Model building completed successfully")
        return trainer
    except Exception as e:
        logger.error(f"Error during model building: {e}")
        raise

def main():
    try:
        mlflow.set_tracking_uri("http://ec2-54-249-36-142.ap-northeast-1.compute.amazonaws.com:5000/") 

        EXPERIMENT_NAME = "BERT_Sentiment_Analysis"
        mlflow.set_experiment(EXPERIMENT_NAME)

        current_dir = os.path.abspath(os.getcwd())
        data_path = os.path.join(current_dir, 'data', 'preprocessed', 'final_data.csv')

        config = load_config('params.yaml')
        df = load_data(data_path)

        X = df['Tweet'].tolist()
        y = df['Sentiment']

        trainer = build_model(config['model_params'], X, y)

        trainer.train()
        trainer.save_model('results/current_model')

        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
    main()