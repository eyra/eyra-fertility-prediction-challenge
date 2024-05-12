import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from catboost import CatBoostClassifier

from submission import clean_df


def train_save_model(cleaned_df: pd.DataFrame, outcome_df: pd.DataFrame, evaluate: bool=False)-> None:
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

    # define the preprocessing pipeline
    features = [c for c in model_df.columns if c not in ['nomem_encr', 'new_child']]
    cat_features = [col for col in features if col.endswith('_ds')]
    num_features = [col for col in features if not col.endswith('_ds')]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])

    # model = VotingClassifier(estimators=[
    #         ('lr', Pipeline(steps=[('preprocessor', preprocessor), ('clf', LogisticRegression(class_weight='balanced', random_state=42))])),
    #         ('rf', Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=300, random_state=42))])),
    #         ('cb', CatBoostClassifier(cat_features=cat_features, verbose=False, random_state=42))
    #     ], voting='hard')
    model = StackingClassifier(
        estimators = [
            ('rf', Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier(n_estimators=300, random_state=42))])),
            ('cb', CatBoostClassifier(cat_features=cat_features, verbose=False, random_state=42))
            ],
        final_estimator = LogisticRegression(class_weight='balanced', random_state=42),
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        )
    
    # fit the model
    model.fit(model_df[features], model_df['new_child'])

    # save the model and params
    joblib.dump(model, Path(__file__).parent / f"model.joblib")

    if evaluate == True:

        print('Performing cross validation...')

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision', 'recall', 'f1']

        # perform cross validation 
        cv_results = cross_validate(model, model_df[features], model_df['new_child'], cv=cv, scoring=scoring)

        # extract metrics from cv_results
        accuracy = cv_results['test_accuracy'].mean()
        precision = cv_results['test_precision'].mean()
        recall = cv_results['test_recall'].mean()
        f1 = cv_results['test_f1'].mean()

        results_df = pd.DataFrame({
            'model_name': ['prefer'], 
            'accuracy': [accuracy],
            'precision': [precision],
            'recall': [recall],
            'f1_score': [f1]})

        # save results
        results_df.to_csv(Path(__file__).parent / 'metrics.csv', index=False)

        # print cv metrics
        print('CV metrics:')
        print(f"\taccuracy: {accuracy:.4f} ({cv_results['test_accuracy'].std():.4f})")
        print(f"\tprecision: {precision:.4f} ({cv_results['test_precision'].std():.4f})")
        print(f"\trecall: {recall:.4f} ({cv_results['test_recall'].std():.4f})")
        print(f"\tf1 score: {f1:.4f} ({cv_results['test_f1'].std():.4f})")        


if __name__ == '__main__':

    proj_dir = Path().cwd()
    parent_proj_dir = proj_dir.parent
    data_dir = parent_proj_dir / 'prefer_data'

    evaluate = False

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
    train_save_model(cleaned_df=cleaned_df, outcome_df=outcome_df, evaluate=evaluate)