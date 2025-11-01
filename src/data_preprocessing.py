import logging
import os
import pandas as pd
import re

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def toLabel(df, column):
    u = df[column].unique()
    mapping = {label: idx for idx, label in enumerate(u)}
    df[column] = df[column].map(mapping)
    return df

def tostring(df, column):
    df[column] = df[column].astype(str)
    return df

def removeHTMLTags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def removeURLs(text):
    url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
    return re.sub(url_pattern, '', text)

def preprocess_text(df):
    try:
        df = toLabel(df, 'Sentiment')
        df = tostring(df, 'Tweet')
        df['Tweet'] = df['Tweet'].apply(removeHTMLTags)
        df['Tweet'] = df['Tweet'].apply(removeURLs)
        logger.info("Text preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Unexpected error during text preprocessing: {e}")
        raise

def save_final_data(df, file_path):
    try:
        path = os.path.join(file_path, 'preprocessed')
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, 'final_data.csv'), index=False)
        logger.info(f"Data saved successfully to {path}")
    except Exception as e:
        logger.error(f"Unexpected error while saving final data: {e}")
        raise

def main():
    try:
        logger.info("Data preprocessing started")

        data = pd.read_csv('data/raw/raw_data.csv')
        logger.info("Raw data loaded successfully")

        preprocessed_data = preprocess_text(data)
        save_final_data(preprocessed_data, 'data')
        logger.info("Data preprocessing finished successfully")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()