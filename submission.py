import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

def clean_df(df, background_df=None):

    # keep rows with available outcome
    df = df[df['outcome_available']==1]

    id_df = df[['nomem_encr']]

    # add golden features
    golden_features = [

        # within how many years do you hope to have your first child 2014-2020
        'cf20m130', # num
        'cf19l130',
        'cf18k130',
        'cf17j130',
        'cf14g130',
        'cf15h130',
        'cf16i130',

        # relationship
        'cf20m031', # num # in what year did you marry 2016-2020
        'cf19l031',
        'cf18k031',
        'cf17j031',
        'cf20m028', # num # when did the relationship start 2020
        'cf17j028', # num # when did the relationship start 2017
        'cf20m029', # num # in what year did you start living together 2020
        'cf19l029', # num # in what year did you start living together 2019
        'cf08a178', # num # partner with previous partners? 2008
        'cf08a179', # num # partner with previous partners? 2008
        'cf10c166', # num # how satisfied with situation as single 2010
        'cf11d186', # num # relationship issues due to working too much 2010

        # children
        'cf20m129', # num # how many more children 2020
        'cf20m456', # num # birth year first child 2020
        'cf20m457', # num # birth year second child 2020
        'cf20m458', # num # birth year third child 2020
        'cf19l472', # num # children passed away 2019

        # childcare - behaviour
        'cf20m203', # cat # getting up from bed at night 2020
        'cf20m204', # cat # washing the child 2020
        'cf20m202', # cat # changing diapers 2020
        'cf20m251', # num # childcare monthly expense 2020

        # childcare - childsitter
        'cf17j398', # num # distance from parents
        'cf20m245', # cat # childsitter yes/no 2020
        'cf20m248', # cat # who is the childsitter 2020
        'cf20m249', # num # how many days per week childsitter
        'cf20m387', # num # childcare supplement
        'cf20m384', # num # subsidies other than child suppleent

        # property
        'cd20m024', # num # property value 2020
        'ca18f023', # num # car value 2018
        'cd20m083', # num # amount mortgage remaining 2020

        # parents
        'cf20m005', # num # year of birth father 2020
        'cf20m009', # num # year of birth mother 2020
    ]
    golden_features_df = df[golden_features + ['nomem_encr']]

    # process background df
    background_df_processed = process_background_df(background_df=background_df, train_df=df, wave_filter=201101)

    # process background df
    bigfive_df = personality_bigfive(train_df=df)

    # merge preprocessed info with train data
    df = pd.merge(id_df, background_df_processed, on='nomem_encr', how='left')
    df = pd.merge(df, golden_features_df, on='nomem_encr', how='left')
    df = pd.merge(df, bigfive_df, on='nomem_encr', how='left')

    # convert numerical features from surveys to string
    list_golden_features_cat = ['cf20m245','cf20m248','cf20m202','cf20m203','cf20m204']
    df[list_golden_features_cat] = df[list_golden_features_cat].astype('str')
    for f in list_golden_features_cat:
        df[f] = df[f].replace('nan', 'missing')
        df.rename(columns = {f : f'{f}_ds'}, inplace = True)

    # create dummy variables for selected features (important and na correlated with response)
    to_dummy_na_list = [
        'cf20m130',
        'cf20m130',
        'cf19l130',
        'cf18k130',
        'cf17j130',
        'cf14g130',
        'cf15h130',
        'cf16i130',
        'cf20m031',
        'cf19l031',
        'cf20m028',
        'cf20m029',
        'cf19l029',
        'cf20m251',
        'cf20m249'
    ]
    for f in to_dummy_na_list:
        df[f'{f}_isna_fl'] = (df[f].isna())*1.0

    # input missing for categorical features
    features = df.columns.tolist()

    cat_features = [col for col in features if col.endswith('_ds')]
    df[cat_features] = df[cat_features].fillna('missing')

    df = df[features]

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # load the model
    model = joblib.load(model_path)

    # preprocess the fake / holdout data
    df = clean_df(df=df, background_df=background_df)

    # exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # return only dataset with predictions and identifier
    return df_predict


### HELPER FUNCTIONS

def process_background_df(background_df, train_df, wave_filter='201601'):
    """
    Process the background DataFrame to extract relevant features for each individual.

    Parameters:
    background_df (DataFrame): The background data.
    train_df (DataFrame): The training data containing the individuals to focus on.
    wave_filter (str, optional): The minimum wave to consider. Defaults to '201601'.

    Returns:
    DataFrame: Aggregated features for each individual.
    """

    # filter for names in training data and after wave filter
    df = background_df[background_df['nomem_encr'].isin(train_df['nomem_encr'])]
    df = df[df['wave'] > wave_filter]
    df = df.sort_values(by='wave')

    # preprocessing functions for each variable
    f_actual = lambda x: x.iloc[-1]                         # get the last value (actual) in the series
    f_med = lambda x: x.dropna().median()                   # get the median value in the series
    f_std = lambda x: x.dropna().std()                      # get the standard deviation of the series
    f_inc = lambda x: (x.max() > x.min())*1.0               # did the series increase value?
    f_inc2 = lambda x: (x.max() > x.min() & x.min()==0)*1.0 # did the series increase value (starting from 0)?

    f_min_equal_one = lambda x: (x.min() == 1)*1.0                    # flag if minimum is equal to one (positie, burgstat)
    f_got_married = lambda x: (x.min() == 1 and x.max() == 5)*1.0     # flag for marriage during obsevation period (burgstat)
    f_map_gender = lambda x: x.map({
        1: 'male',
        2: 'female',
        3: 'Other',
    }).iloc[-1]                                                  # map actual value to string for gender and get the last value
    f_map_civil_status = lambda x: x.map({
        1: 'Married',
        2: 'Separated',
        3: 'Divorced',
        4: 'Widow or widower',
        5: 'Never been married',
    }).iloc[-1]                                                  # map actual value to string for burgstat (civil status) and get the last value
    f_map_domestic_situation = lambda x: x.map({
        1: 'Single',
        2: '(Un)married co-habitation, without child(ren)',
        3: '(Un)married co-habitation, with child(ren)',
        4: ' Single, with child(ren)',
        5: 'Other',
    }).iloc[-1]                                                  # map actual value to string for woonvorm (domestic situation) and get the last value
    f_map_dwelling = lambda x: x.map({
        1: 'Self owned dwelling',
        2: 'Rental dwelling',
        3: 'Sub-rented dwelling',
        4: 'Cost-free dwelling',
        5: 'Unknown (missing)',
    }).iloc[-1]                                                  # map actual value to string for woning (dwelling) and get the last value
    f_map_urban_type = lambda x: x.map({
        1: 'Extremely urban',
        2: 'Very urban',
        3: 'Moderately urban',
        4: 'Slightly urban',
        5: 'Not urban',
    }).iloc[-1]                                                  # map actual value to string for sted (urban type) and get the last value
    f_map_occupation_type = lambda x: x.map({
        1: 'Paid employment',
        2: 'Works or assists in family business',
        3: 'Autonomous professional, freelancer, or self-employed', 
        4: 'Job seeker following job loss', 
        5: 'First-time job seeker', 
        6: 'Exempted from job seeking following job loss',
        7: 'Attends school or is studying',
        8: 'Takes care of the housekeeping',
        9: 'Is pensioner ([voluntary] early retirement, old age pension scheme)',
        10: 'Has (partial) work disability',
        11: 'Performs unpaid work while retaining unemployment benefit',
        12: 'Performs voluntary work',
        13: 'Does something else',
        14: 'Is too young to have an occupation',
    }).iloc[-1]                                                  # map actual value to string for belbezig (occupation type) and get the last value
    
    
    # aggregate using helper functions
    out = df.groupby('nomem_encr').agg({
        'gender_imp': [f_map_gender],                                                      # gender
        'age_imp': [f_actual],                                                             # age
        'partner': [f_actual],                                                             # partner
        'aantalki': [f_actual, f_inc, f_inc2],                                             # number of household children
        'burgstat': [f_got_married, f_min_equal_one, f_map_civil_status],                  # civil status
        'woonvorm': [f_map_domestic_situation],                                            # domestic situation
        'woning': [f_map_dwelling],                                                        # type of dwelling that the household inhabits
        'sted': [f_map_urban_type],                                                        # urban character of place of residence
        'belbezig': [f_map_occupation_type],                                               # primary occupation
        'brutohh_f': [f_actual, f_med, f_std],                                             # gross household income in Euros
    })

    out.columns = out.columns.map('_'.join).str.strip('_')

    out = out.rename(columns={
        'gender_imp_<lambda>': 'gender_ds',
        'age_imp_<lambda>': 'age_qt',
        'partner_<lambda>': 'has_partner_now_fl',
        'aantalki_<lambda_0>': 'actual_number_of_household_children_qt',
        'aantalki_<lambda_1>': 'number_of_household_children_increase_fl',
        'aantalki_<lambda_2>': 'got_first_child_fl',
        'burgstat_<lambda_0>': 'got_married_fl',
        'burgstat_<lambda_1>': 'married_now_fl',
        'burgstat_<lambda_2>': 'actual_civil_status_ds',
        'woonvorm_<lambda>': 'actual_domestic_situation_ds',
        'woning_<lambda>': 'actual_dwelling_ds',
        'sted_<lambda>': 'actual_urban_type_ds',
        'belbezig_<lambda>': 'actual_occupation_type_ds',
        'brutohh_f_<lambda_0>': 'actual_household_gross_monthly_income_qt',
        'brutohh_f_<lambda_1>': 'actual_household_gross_monthly_income_med_qt',
        'brutohh_f_<lambda_2>': 'actual_household_gross_monthly_income_std_qt',
    })

    return out


def personality_bigfive(train_df):
    pattern = r'^cp.*0[2-6][0-9]$'

    codebook_df = pd.read_csv('PreFer_codebook.csv', low_memory=False)
    codebook_df_personality = codebook_df['var_name'][(codebook_df['survey'] == "Personality") & (codebook_df['year'] == 2020)]
    train_personality = train_df[codebook_df_personality]

    five_personality = train_personality.filter(regex=pattern)
    five_pers_inputed = inpute_na(train_df, five_personality.columns.tolist(),codebook_df)

    rows_with_na = five_pers_inputed[five_pers_inputed.isna().all(axis=1)].index
    index_nona = five_pers_inputed[~five_pers_inputed.isna().all(axis=1)].index
    five_pers_inp_nona = five_pers_inputed.loc[index_nona]

    final_personality = five_pers_inp_nona.fillna(five_pers_inp_nona.mean())
    
    scaler =StandardScaler()
    X_scaled=scaler.fit_transform(final_personality)

    fa = FactorAnalyzer(n_factors=5, rotation='oblimin')
    fa.fit(X_scaled)

    factor_scores = fa.transform(X_scaled)
    fa_df = pd.DataFrame(data=factor_scores,columns=['Factor 1','Factor 2','Factor 3','Factor 4','Factor 5'])

    id_nona = train_df['nomem_encr'][index_nona.tolist()].tolist()
    id_nona = pd.DataFrame(id_nona, columns=['nomem_encr']) 
    
    fa_df_tomodel = pd.concat([id_nona, fa_df], axis=1,ignore_index=True)

    id_na = train_df['nomem_encr'][rows_with_na.tolist()].tolist()
    new_rows = pd.DataFrame({'nomem_encr': id_na})
    for col in fa_df_tomodel.columns[1:]:  # Ignora la colonna 'nomem_encr'
        new_rows[col] = None

    fa_df_tomodel = pd.concat([fa_df_tomodel , new_rows], axis=0)
    fa_df_tomodel.loc[fa_df_tomodel.nomem_encr.isna(), 'nomem_encr'] = fa_df_tomodel[0][~fa_df_tomodel[0].isna()]
    fa_df_tomodel = fa_df_tomodel.drop(columns=[0])
    fa_df_tomodel.columns = ['personality_1', 'personality_2', 'personality_3', 'personality_4', 'personality_5', 'nomem_encr']
    
    return fa_df_tomodel


def inpute_na(train_df, var_list, codebook_df):

    out_df = train_df.copy()
    out_df = out_df[var_list]
    print(f'% missing for values for selected variables:\nbefore: {out_df.isna().mean(axis=1).mean():.2%}')

    for var_name in var_list:

        var_label = codebook_df['var_label'][codebook_df ['var_name']==var_name].values[0]
        var_name_hist_codebook = codebook_df[codebook_df ['var_label']==var_label]
        var_name_hist = var_name_hist_codebook.sort_values(by='year', ascending=False)['var_name']

        tmp = train_df[var_name_hist]
        out_df[var_name] = tmp.bfill(axis=1).iloc[:, 0]
        
    print(f'after: {out_df.isna().mean(axis=1).mean():.2%}')
    return out_df