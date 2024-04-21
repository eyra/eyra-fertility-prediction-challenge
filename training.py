import joblib
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from tqdm import tqdm

from submission import clean_df

def train_save_model(cleaned_df: pd.DataFrame, outcome_df: pd.DataFrame)-> None:
    """
    Train and tune a CatBoostClassifier model on the baseline + background features and save it.

    Args:
        cleaned_df (pd.DataFrame): Cleaned data dataframe.
        outcome_df (pd.DataFrame): Dataframe with the outcome variable.
        model_name (str, optional): Name of the model. Defaults to 'baseline'.

    Returns:
        None
    """
    # prepare input data
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")
    features = [c for c in model_df.columns if c not in ['nomem_encr', 'new_child']]
    cat_features = [col for col in features if col.endswith('_ds')]

    # model
    model = CatBoostClassifier(cat_features=cat_features, random_state=42)

    # fit the model
    model.fit(model_df[features], model_df['new_child'], verbose_eval=100)

    # save the model and params
    joblib.dump(model, f"model.joblib")


if __name__ == '__main__':

    data_dir = Path('../prefer_data')

    # import data
    print('Loading data...')
    chunk_list = []
    for chunk in tqdm(pd.read_csv(data_dir / 'training_data/PreFer_train_data.csv', chunksize=1000, low_memory=False)):
        chunk_list.append(chunk)
    raw_df = pd.concat(chunk_list, axis=0)
    
    outcome_df = pd.read_csv(data_dir / 'training_data/PreFer_train_outcome.csv', low_memory=False)
    background_df = pd.read_csv(data_dir / 'other_data/PreFer_train_background_data.csv', low_memory=False)

    # clean data
    print('Cleaning data...')
    cleaned_df = clean_df(df=raw_df, background_df=background_df)
    
    # train and save model
    print(f'Training model...')
    train_save_model(cleaned_df=cleaned_df, outcome_df=outcome_df)