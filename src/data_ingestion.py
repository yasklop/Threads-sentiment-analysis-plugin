import logging
import os
import kagglehub
import yaml
import pandas as pd

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data: {file_path} is empty")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

def preprocess_data(train_df, val_df):
    """Preprocess the data."""
    try:
        df = pd.concat([train_df, val_df], ignore_index=True)
        drop_columns = ['2401', '3364', 'Facebook', 'Irrelevant', "I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣"]
        df = df.drop(columns=drop_columns)
        df = df.rename(columns={'Borderlands': 'Branch', 'Positive': 'Sentiment', 'im getting on borderlands and i will murder you all ,': 'Tweet'})
        df = df[df['Sentiment'] != "Irrelevant"]
        df.dropna(inplace=True)
        df = df.drop_duplicates()
        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def save_data(df, output_path):
    """Save the data to a CSV file."""
    try:
        raw_data_path = os.path.join(output_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        df.to_csv(os.path.join(raw_data_path, "raw_data.csv"), index=False)
        logger.info(f"Data saved successfully to {raw_data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        # Download latest version
        path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
        train_df = load_data(os.path.join(path, "twitter_training.csv"))
        val_df = load_data(os.path.join(path, "twitter_validation.csv"))

        # Preprocess data
        preprocessed_df = preprocess_data(train_df, val_df)

        # Save preprocessed data
        save_data(preprocessed_df, "data")

    except Exception as e:
        logger.error(f"Failed to complete data ingestion: {e}")

if __name__ == "__main__":
    main()
