import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw
df = pd.read_pickle('model_dev1/data/raw/infectious_diseases.pkl')

## get column names
df.columns

## do some data cleaning of column names, 
## make them all lower case, remove white spaces and replace with _ 
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')','')
df.columns

## get data types
df.dtypes
len(df)

# keep columns 
to_keep = [
    'disease',
    'sex',
    'cases',
    'population'
]

df = df[to_keep]
print(df)

## perform ordinal encoding on sex
enc = OrdinalEncoder()
enc.fit(df[['sex']])
df['sex'] = enc.transform(df[['sex']])

## create dataframe with mapping for sex
df_mapping_sex = pd.DataFrame(enc.categories_[0], columns=['sex'])
df_mapping_sex['sex_ordinal'] = df_mapping_sex.index
df_mapping_sex

## save mapping to csv
df_mapping_sex.to_csv('model_dev1/data/processed/mapping_sex.csv', index=False)

## perform ordinal encoding on disease
enc = OrdinalEncoder()
enc.fit(df[['disease']])
df['disease'] = enc.transform(df[['disease']])

## create dataframe with mapping for disease
df_mapping_disease = pd.DataFrame(enc.categories_[0], columns=['disease'])
df_mapping_disease['disease'] = df_mapping_disease.index
df_mapping_disease

## save mapping to csv
df_mapping_disease.to_csv('model_dev1/data/processed/mapping_disease.csv', index=False)

## save a temporary csv file to test the model
df.to_csv('model_dev1/data/processed/infectious_diseases_processed.csv', index=False)