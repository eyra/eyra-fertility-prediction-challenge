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
    
    # def objective(trial):
    
    #     # trial parameters
    #     params = {
    #         'num_trees': trial.suggest_int('num_trees', 100, 500),
    #         'learning_rate' : trial.suggest_float('learning_rate', 0.001, 0.1),
    #         'max_depth' : trial.suggest_int('max_depth', 3, 10),
    #         'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced'])
    #     }

    #     model = CatBoostClassifier(cat_features=cat_features, **params)

    #     # define cross validation and metrics
    #     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    #     # perform cross validation 
    #     cv_results = cross_validate(model, model_df[features], model_df['new_child'], cv=cv, scoring='f1', params={'verbose_eval':500})
    #     score = np.mean(cv_results['test_score'])

    #     return score

    # # create study
    # sampler = optuna.samplers.TPESampler(seed=42)
    # max_trials = 50
    # time_limit = 3600 * 1

    # study = optuna.create_study(
    #     sampler=sampler,
    #     study_name= f'model_optimization',
    #     direction='maximize')

    # # perform optimization
    # print(f'Starting model optimization...')
    # study.optimize(
    #     objective,
    #     n_trials=max_trials,
    #     timeout=time_limit,
    #     show_progress_bar=True
    # )

    # # optimization results
    # print(f"\nNumber of finished trials: {len(study.trials)}")
    # print(f"Best score: {study.best_value}")

    # best_params = study.best_params
    best_params = {
        'num_trees': 328,
        'learning_rate': 0.05592430017269105,
        'max_depth': 7,
        'auto_class_weights': 'Balanced'
    }

    # model
    model = CatBoostClassifier(cat_features=cat_features, random_state=42, **best_params)

    # fit the model
    model.fit(model_df[features], model_df['new_child'], verbose_eval=100)

    # save the model and params
    joblib.dump(model, f"model.joblib")
    joblib.dump(best_params, f"model__best_params.joblib")


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