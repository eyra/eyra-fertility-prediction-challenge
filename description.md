# Description of submission

## Modeling Procedure

This project employs a machine learning approach to predict the likelihood of an individual having a child during 2021-2023 using LISS data. The modeling procedure involves the following steps:

- data preparation:
    - clean and preproce the raw data
    - extract relevant background variables for each individual
    - Aggregat values for each individual over the years

- modeling:
    - train a CatBoostClassifier model on the prepared data
    - use the model to predict the outcome variable (i.e., whether an individual had a child during 2021-2023)