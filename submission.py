import pandas as pd
import joblib


def clean_df(df, background_df=None):

    # keep rows with available outcome
    df = df[df['outcome_available']==1]

    # process background df
    background_df_processed = process_background_df(background_df=background_df, train_df=df, wave_filter=201101)
    background_features = [col for col in background_df_processed.columns if not col.endswith('_ds')]
    background_df_processed = background_df_processed[background_features]

    # merge with trai data
    df = pd.merge(df, background_df_processed, on='nomem_encr', how='left')

    # select variables for modelling
    features = [

        # id
        "nomem_encr",

        # background variables
        'gender_qt',
        'age_qt',
        'origin_qt',
        
        'is_household_lead_fl',
        'max_position_within_the_household_qt',
        'household_head_age_qt',
        'actual_number_of_household_members_qt',
        'min_number_of_household_members_qt',
        'max_number_of_household_members_qt',
        'number_of_household_members_increase_fl',
        'actual_number_of_household_children_qt',
        'min_number_of_household_children_qt',
        'max_number_of_household_children_qt',
        'number_of_household_children_increase_fl',
        'has_partner_now_fl',
        'got_married_fl',
        'married_now_fl',
        'actual_civil_status_qt',
        
        'actual_domestic_situation_qt',
        'actual_dwelling_qt',
        'actual_urban_type_qt',
        'actual_occupation_type_qt',
        'actual_personal_gross_monthly_income_qt',
        'actual_personal_gross_monthly_income_med_qt',
        'actual_personal_gross_monthly_income_std_qt',
        'actual_personal_gross_monthly_income_descr_qt',
        'actual_personal_net_monthly_income_qt',
        'actual_personal_net_monthly_income_med_qt',
        'actual_personal_net_monthly_income_std_qt',
        'actual_personal_net_monthly_income_descr_qt',
        'actual_household_gross_monthly_income_qt',
        'actual_household_gross_monthly_income_med_qt',
        'actual_household_gross_monthly_income_std_qt',
        'actual_household_net_monthly_income_qt',
        'actual_household_net_monthly_income_med_qt',
        'actual_household_net_monthly_income_std_qt',
        'actual_education_qt',
        'got_degree_fl'
    ] 

    # keep data with variables selected
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
    train_df_filter = train_df[train_df['outcome_available'] == 1]
    df = background_df[background_df['nomem_encr'].isin(train_df_filter['nomem_encr'])]
    df = df[df['wave'] > wave_filter]
    df = df.sort_values(by='wave')

    # preprocessing functions for each variable
    f_actual = lambda x: x.iloc[-1]                         # get the last value (actual) in the series
    f_min = lambda x: x.min()                               # get the minimum value in the series
    f_max = lambda x: x.max()                               # get the maximum value in the series
    f_med = lambda x: x.dropna().median()                   # get the median value in the series
    f_std = lambda x: x.dropna().std()                      # get the standard deviation of the series
    f_inc = lambda x: (x.max() > x.min())*1.0               # did the series increase value?
    f_educ = lambda x: (x.max() == 6 and x.min() < 6)*1.0   # did the suject get a degree?

    f_min_equal_one = lambda x: (x.min() == 1)*1.0                    # flag if minimum is equal to one (positie, burgstat)
    f_got_married = lambda x: (x.min() == 1 and x.max() == 5)*1.0     # flag for marriage during obsevation period (burgstat)
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
    f_map_income = lambda x: x.map({
        0: 'No income',
        1: '500 euros or less',
        2: '501 to 1000 euros',
        3: '1001 to 1500 euros', 
        4: '1501 to 2000 euros', 
        5: '2001 to 2500 euros', 
        6: '2501 to 3000 euros',
        7: '3001 to 3500 euros',
        8: '3501 to 4000 euros',
        9: '4001 to 4500 euros',
        10: '4501 to 5000 euros',
        11: '5001 to 7500 euros',
        12: 'More than 7500 euros',
        13: "I don't know",
        14: 'I prefer not to say'
    }).iloc[-1]                                                  # map actual value to string for brutoink_f and nettoink_f (gross and net income) and get the last value
    f_map_education = lambda x: x.map({
        1: 'primary school',
        2: 'vmbo (intermediate secondary education, US: junior high school)',
        3: 'havo/vwo (higher secondary education/preparatory university education, US:senior high school)',
        4: 'mbo (intermediate vocational education, US: junior college)',
        5: 'hbo (higher vocational education, US: college)',
        6: 'wo (university)',
        7: 'other',
        8: 'Not (yet) completed any education',
        9: 'Not yet started any education'
    }).iloc[-1]                                                  # map actual value to string for oplmet (education level) and get the last value
    f_map_origin = lambda x: x.map({
        0: 'Dutch background',
        101: 'First generation foreign, Western background',
        102: 'First generation foreign, non-western background',
        201: 'Second generation foreign, Western background',
        202: 'Second generation foreign, non-western background',
        999: 'Origin unknown or part of the information unknown (missing values)'
    }).iloc[-1]                                                  # map actual value to string for migration_background_imp (origin) and get the last value
    
    # aggregate using helper functions
    out = df.groupby('nomem_encr').agg({
        'gender_imp': [f_actual],                                                          # gender
        'age_imp': [f_actual],                                                             # age
        'migration_background_imp': [f_actual, f_map_origin],                              # origin
        'positie': [f_min_equal_one, f_max],                                               # position within the household
        'lftdhhh': [f_actual],                                                             # age of the household head
        'aantalhh': [f_actual, f_min, f_max, f_inc],                                       # number of household members
        'aantalki': [f_actual, f_min, f_max, f_inc],                                       # number of household children
        'partner': [f_max],                                                                # the household head lives together with a partner
        'burgstat': [f_got_married, f_min_equal_one, f_actual, f_map_civil_status],        # civil status
        'woonvorm': [f_actual, f_map_domestic_situation],                                  # Domestic situation
        'woning': [f_actual, f_map_dwelling],                                              # type of dwelling that the household inhabits
        'sted': [f_actual, f_map_urban_type],                                              # urban character of place of residence
        'belbezig': [f_actual, f_map_occupation_type],                                     # primary occupation
        'brutoink_f': [f_actual, f_med, f_std],                                            # personal gross monthly income in Euros, imputed
        'brutocat': [f_actual, f_map_income],                                              # personal gross monthly income in categories
        'nettoink_f': [f_actual, f_med, f_std],                                            # personal net monthly income in Euros, imputed
        'nettocat': [f_actual, f_map_income],                                              # personal net monthly income in categories
        'brutohh_f': [f_actual, f_med, f_std],                                             # gross household income in Euros
        'nettohh_f': [f_actual, f_med, f_std],                                             # net household income in Euros
        # 'oplzon': [f_actual, f_map_education],                                           # highest level of education irrespective of diploma
        'oplmet': [f_actual, f_map_education, f_educ],                                     # highest level of education with diploma
        # 'oplcat': [f_actual, f_map_education],                                           # level of education in CBS (Statistics Netherlands) categories
    })

    out.columns = out.columns.map('_'.join).str.strip('_')

    out = out.rename(columns={
        'gender_imp_<lambda>': 'gender_qt',
        'age_imp_<lambda>': 'age_qt',
        'migration_background_imp_<lambda_0>': 'origin_qt',
        'migration_background_imp_<lambda_1>': 'origin_ds',
        'positie_<lambda_0>': 'is_household_lead_fl',
        'positie_<lambda_1>': 'max_position_within_the_household_qt',
        'lftdhhh_<lambda>': 'household_head_age_qt',
        'aantalhh_<lambda_0>': 'actual_number_of_household_members_qt',
        'aantalhh_<lambda_1>': 'min_number_of_household_members_qt',
        'aantalhh_<lambda_2>': 'max_number_of_household_members_qt',
        'aantalhh_<lambda_3>': 'number_of_household_members_increase_fl',
        'aantalki_<lambda_0>': 'actual_number_of_household_children_qt',
        'aantalki_<lambda_1>': 'min_number_of_household_children_qt',
        'aantalki_<lambda_2>': 'max_number_of_household_children_qt',
        'aantalki_<lambda_3>': 'number_of_household_children_increase_fl',
        'partner_<lambda>': 'has_partner_now_fl',
        'burgstat_<lambda_0>': 'got_married_fl',
        'burgstat_<lambda_1>': 'married_now_fl',
        'burgstat_<lambda_2>': 'actual_civil_status_qt',
        'burgstat_<lambda_3>': 'actual_civil_status_ds',
        'woonvorm_<lambda_0>': 'actual_domestic_situation_qt',
        'woonvorm_<lambda_1>': 'actual_domestic_situation_ds',
        'woning_<lambda_0>': 'actual_dwelling_qt',
        'woning_<lambda_1>': 'actual_dwelling_ds',
        'sted_<lambda_0>': 'actual_urban_type_qt',
        'sted_<lambda_1>': 'actual_urban_type_ds',
        'belbezig_<lambda_0>': 'actual_occupation_type_qt',
        'belbezig_<lambda_1>': 'actual_occupation_type_ds',
        'brutoink_f_<lambda_0>': 'actual_personal_gross_monthly_income_qt',
        'brutoink_f_<lambda_1>': 'actual_personal_gross_monthly_income_med_qt',
        'brutoink_f_<lambda_2>': 'actual_personal_gross_monthly_income_std_qt',
        'brutocat_<lambda_0>': 'actual_personal_gross_monthly_income_descr_qt',
        'brutocat_<lambda_1>': 'actual_personal_gross_monthly_income_descr_ds',
        'nettoink_f_<lambda_0>': 'actual_personal_net_monthly_income_qt',
        'nettoink_f_<lambda_1>': 'actual_personal_net_monthly_income_med_qt',
        'nettoink_f_<lambda_2>': 'actual_personal_net_monthly_income_std_qt',
        'nettocat_<lambda_0>': 'actual_personal_net_monthly_income_descr_qt',
        'nettocat_<lambda_1>': 'actual_personal_net_monthly_income_descr_ds',
        'nettohh_f_<lambda_0>': 'actual_household_net_monthly_income_qt',
        'nettohh_f_<lambda_1>': 'actual_household_net_monthly_income_med_qt',
        'nettohh_f_<lambda_2>': 'actual_household_net_monthly_income_std_qt',
        'brutohh_f_<lambda_0>': 'actual_household_gross_monthly_income_qt',
        'brutohh_f_<lambda_1>': 'actual_household_gross_monthly_income_med_qt',
        'brutohh_f_<lambda_2>': 'actual_household_gross_monthly_income_std_qt',
        'oplmet_<lambda_0>': 'actual_education_qt',
        'oplmet_<lambda_1>': 'actual_education_ds',
        'oplmet_<lambda_2>': 'got_degree_fl',
    })

    return out