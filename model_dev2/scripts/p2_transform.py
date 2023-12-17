import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## get data raw

datalink = 'https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)

## get column names
df.columns

## data cleaning
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## get data types
df.dtypes 
len(df)

# changing these columns from float64 to object
df['addr_pct_cd']= df['addr_pct_cd'].astype(object)
df['jurisdiction_code'] = df['jurisdiction_code'].astype(object)
df['transit_district'] = df['transit_district'].astype(object)
df['vic_age_group'] = df['vic_age_group'].astype(object)

## drop columns
to_drop = [
    'cmplnt_num',
    'addr_pct_cd',
    'cmplnt_fr_dt',
    'cmplnt_fr_tm',
    'cmplnt_to_tm',
    'cmplnt_to_dt',
    'crm_atpt_cptd_cd',
    'hadevelopt',
    'housing_psa',
    'jurisdiction_code',
    'ky_cd',
    'parks_nm',
    'patrol_boro',
    'pd_cd',
    'pd_desc',
    'prem_typ_desc',
    'rpt_dt',
    'station_name',
    'transit_district',
    'x_coord_cd',
    'y_coord_cd',
    'latitude',
    'longitude',
    'lat_lon',
    'new_georeferenced_column',
    'susp_age_group',
    'susp_race',
    'susp_sex',
    'vic_sex'
]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')

## Drop missing values
df = df.replace('(null)', pd.NA)
df = df.dropna(axis=0, how='any', subset=None)


## boro_nm --> will need to encode this
df.boro_nm.value_counts()

## perform ordinal encoding on boro_nm
enc = OrdinalEncoder()
enc.fit(df[['boro_nm']])
df['boro_nm'] = enc.transform(df[['boro_nm']])

## create dataframe with mapping
df_mapping_nycboro = pd.DataFrame(enc.categories_[0], columns=['boro_nm'])
df_mapping_nycboro['boro_nm_ordinal'] = df_mapping_nycboro.index
df_mapping_nycboro.head(5)

## save mapping to csv
df_mapping_nycboro.to_csv('model_dev2/data/processed/mapping_nycboro.csv', index=False)



## juris_desc --> will need to encode this
df.juris_desc.value_counts()

## perform ordinal encoding on juris_desc
enc = OrdinalEncoder()
enc.fit(df[['juris_desc']])
df['juris_desc'] = enc.transform(df[['juris_desc']])

## create dataframe with mapping
df_mapping_jurisdiction = pd.DataFrame(enc.categories_[0], columns=['juris_desc'])
df_mapping_jurisdiction['juris_desc_ordinal'] = df_mapping_jurisdiction.index
df_mapping_jurisdiction.head(5)

# save mapping to csv
df_mapping_jurisdiction.to_csv('model_dev2/data/processed/mapping_jurisdiction.csv', index=False)



## law_cat_cd --> will need to encode this
df.law_cat_cd.value_counts()

## perform orindla encoding on law_cat_cd
enc = OrdinalEncoder()
enc.fit(df[['law_cat_cd']])
df['law_cat_cd'] = enc.transform(df[['law_cat_cd']])

## create dataframe with mapping
df_mapping_law_code = pd.DataFrame(enc.categories_[0], columns=['law_cat_cd'])
df_mapping_law_code['law_cat_cd_ordinal'] = df_mapping_law_code.index
df_mapping_law_code.head(5)

# save mapping to csv
df_mapping_law_code.to_csv('model_dev2/data/processed/mapping_law_code.csv', index=False)



## loc_of_occur_desc --> will need to encode this
df.loc_of_occur_desc.value_counts()

## perform orindla encoding on loc_of_occur_desc
enc = OrdinalEncoder()
enc.fit(df[['loc_of_occur_desc']])
df['loc_of_occur_desc'] = enc.transform(df[['loc_of_occur_desc']])

## create dataframe with mapping
df_mapping_location_occurrence = pd.DataFrame(enc.categories_[0], columns=['loc_of_occur_desc'])
df_mapping_location_occurrence['loc_of_occur_desc_ordinal'] = df_mapping_location_occurrence.index
df_mapping_location_occurrence.head(5)

# save mapping to csv
df_mapping_location_occurrence.to_csv('model_dev2/data/processed/mapping_location.csv', index=False)



## ofns_desc --> will need to encode this
df.ofns_desc.value_counts()

## perform orindla encoding on ofns_desc
enc = OrdinalEncoder()
enc.fit(df[['ofns_desc']])
df['ofns_desc'] = enc.transform(df[['ofns_desc']])

## create dataframe with mapping
df_mapping_offense = pd.DataFrame(enc.categories_[0], columns=['ofns_desc'])
df_mapping_offense['ofns_desc_ordinal'] = df_mapping_offense.index
df_mapping_offense.head(5)

# save mapping to csv
df_mapping_offense.to_csv('model_dev2/data/processed/mapping_offense.csv', index=False)




## vic_age_group --> will need to encode this
df.vic_age_group.value_counts()

## perform orindla encoding on vic_age_group
enc = OrdinalEncoder()
enc.fit(df[['vic_age_group']])
df['vic_age_group'] = enc.transform(df[['vic_age_group']])

## create dataframe with mapping
df_mapping_victim_age = pd.DataFrame(enc.categories_[0], columns=['vic_age_group'])
df_mapping_victim_age['vic_age_group_ordinal'] = df_mapping_victim_age.index
df_mapping_victim_age.head(5)

# save mapping to csv
df_mapping_victim_age.to_csv('model_dev2/data/processed/mapping_victim_age.csv', index=False)



## vic_race --> will need to encode this
df.vic_race.value_counts()

## perform orindla encoding on vic_race
enc = OrdinalEncoder()
enc.fit(df[['vic_race']])
df['vic_race'] = enc.transform(df[['vic_race']])

## create dataframe with mapping
df_mapping_victim_race = pd.DataFrame(enc.categories_[0], columns=['vic_race'])
df_mapping_victim_race['vic_race_ordinal'] = df_mapping_victim_race.index
df_mapping_victim_race.head(5)

# save mapping to csv
df_mapping_victim_race.to_csv('model_dev2/data/processed/mapping_victim_race.csv', index=False)



# save temporary csv files to test the model
df.head(10000).to_csv('model_dev2/data/processed/complaint_10k.csv', index=False)
df.sample(50000).to_csv('model_dev2/data/processed/complaint_50k.csv', index=False)
df.sample(100000).to_csv('model_dev2/data/processed/complaint_100k.csv', index=False)