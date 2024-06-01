import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

def clean_df(df, background_df=None):

    # keep rows with available outcome
    df = df[df['outcome_available']==1]

    id_df = df[['nomem_encr']]

    feature_2020_impute = [

        # Family & Household
        'cf20m130', # within how many years do you hope to have your first child
        'cf20m129', # How many more children
        'cf20m128', # Do you want another child ?
        
        'cf20m005', # Year of birth father 
        'cf20m008', # When did yout father pass away # 
        'cf20m009', # Year of birth mum #
        'cf20m012', # When did yuor mum pass away #  
        'cf20m014', # How old were you when your parents separated
        'cf20m398', # Distance from parents

        'cf20m025', # Living with partner
        'cf20m028', # Relationship start year
        'cf20m029', # Living together start year
        'cf20m030', # Are you maried
        'cf20m031', # In what year did you marry
        'cf20m402', # Same partner
        'cf20m032', # gender partner
        'cf20m166', # How satisfied with situation as single
        'cf20m185', # Partner disagreement frequency
        'cf20m186', # Relationship issues due to working too much

        'cf20m250', # Childcare usage
        'cf20m251', # Childcare monthly expense
        
        'cf20m456', # First child's birth year
        'cf20m457', # Second child's birth year
        'cf20m458', # Third child's birth year
        'cf20m459', # Fourth child's birth year
        'cf20m460', # Fifth child's birth year
        'cf20m461', # Sixth child's birth year
        'cf20m462', # Seventh child's birth year
        'cf20m463', # Eighth child's birth year
        'cf20m464', # Ninth child's birth year
        'cf20m465', # Tenth child's birth year
        'cf20m466', # Eleventh child's birth year
        'cf20m467', # Twelfth child's birth year
        'cf20m468', # Thirteenth child's birth year
        'cf20m469', # Fourteenth child's birth year
        'cf20m470', # Fifteenth child's birth year

        'cf20m471', # Children passed away
        'cf20m472', # What age Children passed away
        'cf20m486', # Household chores division

        # Politics and Values 
        'cv20l068', # Political views
        'cv20l103', # Overall satisfaction
        'cv20l125', # Marriage and children
        'cv20l126', # One parent vs two
        'cv20l130', # Divorce normalcy

        # Social Integration and Leisure
        'cs20m180', # Leisure time hours
        'cs20m370', # Education level

        # Religion and Ethnicity
        'cr20m041',  # Religiosity    
        'cr20m093', # Speaking Dutch with partner
        'cr20m094', # Speaking Dutch with children

        # Economic Situation Assets
        'ca20g023', # car value
        # Economic Situation Housing
        'cd20m024',  # property value 2020
        # Economic Situation Income
        'ci20m043', # Work situation
        'ci20m309', # Paying for children's expenses (multiple answer)

        # Health
        'ch20m219', # gynaecologist
    ]

    features_background_impute = [        
        # Background
        'belbezig_2020', # occupation
        'brutoink_f_2020', # gross income
        'nettoink_f_2020', # net income
        'burgstat_2020', # civil status
        'oplcat_2020', # education
        'partner_2020', # lives with partner
        'sted_2020', # urban type
        'woning_2020', # dwelling type
        'woonvorm_2020' # domestic situation
        ]

    feature_2020_notimpute = [

        # Family & Household
        'cf20m130', # within how many years do you hope to have your first child
        'cf20m129', # How many more children
        'cf20m128', # Do you want another child ?
        
        'cf20m025', # Living with partner
        'cf20m030', # Are you maried
        'cf20m402', # Same partner
        'cf20m166', # How satisfied with situation as single

        # Health
        'ch20m219', # gynaecologist

        # Background
        'migration_background_bg',
        'belbezig_2020', # occupation
        'brutoink_f_2020', # gross income
        'nettoink_f_2020', # net income
        'burgstat_2020', # civil status
        'oplcat_2020', # education
        'partner_2020', # lives with partner
        'sted_2020', # urban type
        'woning_2020', # dwelling type
        'woonvorm_2020' # domestic situation
    ]

    child_birth_years = [        
        'cf20m456', # First child's birth year
        'cf20m457', # Second child's birth year
        'cf20m458', # Third child's birth year
        'cf20m459', # Fourth child's birth year
        'cf20m460', # Fifth child's birth year
        'cf20m461', # Sixth child's birth year
        'cf20m462', # Seventh child's birth year
        'cf20m463', # Eighth child's birth year
        'cf20m464', # Ninth child's birth year
        'cf20m465', # Tenth child's birth year
        'cf20m466', # Eleventh child's birth year
        'cf20m467', # Twelfth child's birth year
        'cf20m468', # Thirteenth child's birth year
        'cf20m469', # Fourteenth child's birth year
        'cf20m470', # Fifteenth child's birth year
        ]
    
    # imputation 
    codebook_df = pd.read_csv('PreFer_codebook.csv', low_memory=False)
    df_impute_noback = pd.merge(df[['nomem_encr']], inpute_na(df, feature_2020_impute, codebook_df, method = ''), left_index=True, right_index=True)
    df_impute_back = pd.merge(df[['nomem_encr']], inpute_na(df, features_background_impute, codebook_df), left_index=True, right_index=True)
    df_impute = pd.merge(df_impute_back, df_impute_noback, on = ['nomem_encr'], how = 'inner')
    
    # year last child, how many children
    df_impute['year_last_child'] = df_impute[child_birth_years].max(axis=1, skipna=True)
    df_impute['num_children'] = df_impute[child_birth_years].notna().sum(axis=1)
    df_impute.drop(columns=child_birth_years, inplace=True)

    # add raw features 
    for c in df_impute.columns:
        if c != 'nomem_encr':
            df_impute.rename(columns={c: f'{c}_imputed'}, inplace=True)

    df_new = pd.merge(df_impute, df[feature_2020_notimpute+['nomem_encr']],  on = 'nomem_encr', how = 'inner')

    feature_super_gold  =  [    
        # within how many years do you hope to have your first child
        'cf20m130', # 2020
        'cf19l130', # 2019
        'cf18k130', # 2018
        'cf17j130', # 2017
        'cf16i130', # 2016
        'cf15h130', # 2015
        'cf14g130', # 2014
        'cf13f130', # 2013
        'cf12e130', # 2012
        'cf11d130', # 2011
        'cf09b130', # 2009
        'cf08a130', # 2008
        ]   

    df_zero = imputation_cf20_130(feature_super_gold, train_df=df)
    df_negative = imputation_cf20_130_negative(feature_super_gold, train_df=df)
    df_super_gold_imputed = pd.merge(df_zero, df_negative, on = 'nomem_encr', how = 'inner')
    df2 = pd.merge(df_super_gold_imputed, df_new, on = 'nomem_encr', how = 'inner')

    # process background df
    background_df_processed = process_background_df(background_df=background_df, train_df=df, wave_filter=201101)
    background_gold = [
        'actual_household_gross_monthly_income_qt',
        'actual_household_net_monthly_income_qt',
        'actual_household_gross_monthly_income_med_qt',
        'actual_household_gross_monthly_income_std_qt',
        'actual_household_net_monthly_income_std_qt',
        'age_qt',
        'gender_ds',
        'got_married_fl',
        'actual_household_net_monthly_income_med_qt']
    background_df_processed = background_df_processed[background_gold]

    # same sex
    df2['cf20m032_imputed'] = df2['cf20m032_imputed'].replace({1: 'male', 2: 'female'})
    df3 = pd.merge(df2, background_df_processed['gender_ds'], on = 'nomem_encr', how='left')
    df3['same_sex_ds'] = df3['cf20m032_imputed'] == df3['gender_ds']
    df3.drop(columns=['cf20m032_imputed', 'gender_ds'], inplace=True)

    # big five
    bigfive_df = personality_bigfive(train_df=df)

    # merge preprocessed info with train data
    df = pd.merge(id_df, background_df_processed, on='nomem_encr', how='left')
    df = pd.merge(df, df3, on='nomem_encr', how='left')
    df = pd.merge(df, bigfive_df, on='nomem_encr', how='left')

    cat_features = [col for col in df.columns.tolist() if col.endswith('_ds')]
    df[cat_features] = df[cat_features].fillna('missing')
    for c in cat_features:
        df[c] = df[c].astype('category')

    features = df.columns.tolist()
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

    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # load the model
    models = joblib.load(model_path)

    # preprocess the fake / holdout data
    df = clean_df(df=df, background_df=background_df)

    # exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']

    # generate predictions from model, should be 0 (no child) or 1 (had child)
    et_preds = models['et'].predict_proba(df[vars_without_id])[:, 1]
    cb_preds = models['cb'].predict_proba(df[vars_without_id])[:, 1]
    lgb_preds = models['lgb'].predict_proba(df[vars_without_id])[:, 1]

    # average prediction for class 1
    final_preds = cb_preds*0.5 + et_preds*0.25 + lgb_preds*0.25
    predictions = final_preds.round()

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
        'nettohh_f': [f_actual, f_med, f_std],                                             # gross household income in Euros

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
        'nettohh_f_<lambda_0>': 'actual_household_net_monthly_income_qt',
        'nettohh_f_<lambda_1>': 'actual_household_net_monthly_income_med_qt',
        'nettohh_f_<lambda_2>': 'actual_household_net_monthly_income_std_qt',
    })

    return out


def personality_bigfive(train_df):
    pattern = r'^cp.*0[2-6][0-9]$'

    codebook_df = pd.read_csv('PreFer_codebook.csv', low_memory=False)
    codebook_df.head()
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


def inpute_na(train_df, var_list, codebook_df, method='var_label'):

    out_df = train_df.copy()
    out_df = out_df[var_list]
    print(f'% missing for values for selected variables:\nbefore: {out_df.isna().mean(axis=1).mean():.2%}')

    for var_name in var_list:
        survey_tmp = codebook_df.survey[codebook_df.var_name == var_name].values[0]
        if method=='var_label':
            var_label = codebook_df['var_label'][codebook_df ['var_name']==var_name].values[0]
            var_name_hist_codebook = codebook_df[codebook_df ['var_label']==var_label]
            var_name_hist_codebook = var_name_hist_codebook.loc[var_name_hist_codebook.survey.str.contains(survey_tmp),:]

        else:
            var_name_hist_codebook = codebook_df[codebook_df['var_name'].str.startswith(var_name[:2]) & codebook_df['var_name'].str.endswith(var_name[-3:])]
        
        var_name_hist = var_name_hist_codebook.sort_values(by='year', ascending=False)['var_name']
        
        tmp = train_df[var_name_hist]
        out_df[var_name] = tmp.bfill(axis=1).iloc[:, 0]
        
    print(f'after: {out_df.isna().mean(axis=1).mean():.2%}')
    return out_df


def imputation_cf20_130(list_features, train_df):

    prova_train = train_df[list_features + ['nomem_encr']]

    new_cf20m130 = []
    indici_salvati = prova_train['nomem_encr']
    prova_prova_train = prova_train.fillna(-1000).drop(columns=['nomem_encr'])
    valore_nan = -1000

    for i in prova_prova_train.index:
        if prova_prova_train['cf20m130'][i] == valore_nan:
            count_col = 0
            for col in prova_prova_train.columns[::-1]:
                count_col = count_col + 1
                if prova_prova_train[col][i] != valore_nan:
                    new_value = prova_prova_train[col][i] - (count_col-1)
                    if new_value > 0:
                        new_cf20m130.append([indici_salvati[i],new_value])
                    else:
                        new_cf20m130.append([indici_salvati[i],0])
                    break
                if count_col == prova_prova_train.shape[1]:
                    new_cf20m130.append([indici_salvati[i],float('nan')])
        else:
            if prova_prova_train['cf20m130'][i]>2000:
                new_cf20m130.append([indici_salvati[i],prova_prova_train['cf20m130'][i]-2020])
            else:
                new_cf20m130.append([indici_salvati[i],prova_prova_train['cf20m130'][i]])

                
    new_cf20m130 = pd.DataFrame(new_cf20m130, columns=['nomem_encr','cf20m130_zero'])
    
    return new_cf20m130


def imputation_cf20_130_negative(list_features, train_df):

    prova_train = train_df[list_features + ['nomem_encr']]

    new_cf20m130 = []
    indici_salvati = prova_train['nomem_encr']
    prova_prova_train = prova_train.fillna(-1000).drop(columns=['nomem_encr'])
    valore_nan = -1000

    for i in prova_prova_train.index:
        if prova_prova_train['cf20m130'][i] == valore_nan:
            count_col = 0
            for col in prova_prova_train.columns[::-1]:
                count_col = count_col + 1
                if prova_prova_train[col][i] != valore_nan:
                    new_value = prova_prova_train[col][i] - (count_col-1)
                    # if new_value > 0:
                    new_cf20m130.append([indici_salvati[i],new_value])
                    # else:
                    #     new_cf20m130.append([indici_salvati[i],0])
                    break
                if count_col == prova_prova_train.shape[1]:
                    new_cf20m130.append([indici_salvati[i],float('nan')])
        else:
            if prova_prova_train['cf20m130'][i]>2000:
                new_cf20m130.append([indici_salvati[i],prova_prova_train['cf20m130'][i]-2020])
            else:
                new_cf20m130.append([indici_salvati[i],prova_prova_train['cf20m130'][i]])
                
    new_cf20m130 = pd.DataFrame(new_cf20m130, columns=['nomem_encr','cf20m130_negative'])
    
    return new_cf20m130