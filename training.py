import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from submission import clean_df


def train_save_model(cleaned_df: pd.DataFrame, outcome_df: pd.DataFrame, evaluate: bool=False, tune: bool=False)-> None:
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

        # prepare input data
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # define the preprocessing pipeline
    features = [c for c in model_df.columns if c not in ['nomem_encr', 'new_child']]
    cat_features = [col for col in features if col.endswith('_ds')]
    num_features = [col for col in features if not col.endswith('_ds')]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])
    
    params_catboost = {
        'iterations': 1000,
        'learning_rate': 0.04465495649788828,
        'depth': 7,
        'subsample': 0.5990716998946282, 
        'colsample_bylevel': 0.15856264300042117, 
        'min_data_in_leaf': 48
        }
    params_lgb = {
        'bagging_fraction': 0.8, 
        'feature_fraction': 0.9, 
        'learning_rate': 0.1, 
        'max_bin': 20, 
        'max_depth': 30, 
        'min_data_in_leaf': 20, 
        'min_sum_hessian_in_leaf': 0.001, 
        'n_estimators': 3246, 
        'num_leaves': 24, 
        'subsample': 1.0
        }

    models = {
        'et': Pipeline(steps=[('preprocessor', preprocessor), ('clf', ExtraTreesClassifier(n_estimators=500, max_features=0.3, random_state=42))]),
        'cb': CatBoostClassifier(cat_features=cat_features, verbose=False, random_state=42, **params_catboost),
        'lgb':  LGBMClassifier(boosting_type = 'gbdt', random_state=42, verbose=-1, class_weight = 'balanced', **params_lgb)
    }

    # fit the model
    models['et'].fit(model_df[features], model_df['new_child'])
    models['cb'].fit(model_df[features], model_df['new_child'])
    models['lgb'].fit(model_df[features], model_df['new_child'], categorical_feature=cat_features)

    # save the model and params
    joblib.dump(models, Path(__file__).parent / f"model.joblib")

    if evaluate == True:

        print('\nPerforming cross validation...')

        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=10, random_state=42)
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=1927)
        cv_results = []

        for train_index, test_index in tqdm(cv.split(model_df[features], model_df['new_child'])):
            X_train, X_test = model_df.iloc[train_index][features], model_df.iloc[test_index][features]
            y_train, y_test = model_df.iloc[train_index]['new_child'], model_df.iloc[test_index]['new_child']

            # fit models
            models['et'].fit(X_train, y_train)
            models['cb'].fit(X_train, y_train)
            models['lgb'].fit(X_train, y_train, categorical_feature=cat_features)

            # predictions
            et_preds = models['et'].predict_proba(X_test)[:, 1]
            cb_preds = models['cb'].predict_proba(X_test)[:, 1]
            lgb_preds = models['lgb'].predict_proba(X_test)[:, 1]

            # average prediction for class 1
            final_preds = cb_preds*0.5 + et_preds*0.25 + lgb_preds*0.25

            # metrics
            acc = accuracy_score(y_test, final_preds.round())
            prec = precision_score(y_test, final_preds.round())
            rec = recall_score(y_test, final_preds.round())
            f1 = f1_score(y_test, final_preds.round())

            cv_results.append([acc, prec, rec, f1])
        
        # extract metrics from cv_results    
        accuracy_scores = [result[0] for result in cv_results]
        precision_scores = [result[1] for result in cv_results]
        recall_scores = [result[2] for result in cv_results]
        f1_scores = [result[3] for result in cv_results]

        accuracy = np.mean(accuracy_scores)
        precision = np.mean(precision_scores)
        recall = np.mean(recall_scores)
        f1 = np.mean(f1_scores)

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
        print(f"\taccuracy: {accuracy:.4f} ({np.std(accuracy_scores):.4f})")
        print(f"\tprecision: {precision:.4f} ({np.std(precision_scores):.4f})")
        print(f"\trecall: {recall:.4f} ({np.std(recall_scores):.4f})")
        print(f"\tf1 score: {f1:.4f} ({np.std(f1_scores):.4f})")      


if __name__ == '__main__':

    proj_dir = Path().cwd()
    parent_proj_dir = proj_dir.parent
    data_dir = parent_proj_dir / 'prefer_data'

    evaluate = True

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