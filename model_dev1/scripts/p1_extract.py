import pandas as pd 

## get data 

# data download link: 
datalink = 'https://data.chhs.ca.gov/dataset/03e61434-7db8-4a53-a3e2-1d4d36d6848d/resource/75019f89-b349-4d5e-825d-8b5960fc028c/download/odp_idb_2001_2022_ddg_compliant.csv'

df = pd.read_csv(datalink)
df.size
df.sample(5)


## save as csv to model_dev1/data/raw/
df.to_csv('model_dev1/data/raw/infectious_diseases.csv', index=False)

## save as pickle to model_dev1/data/raw/
df.to_pickle('model_dev1/data/raw/infectious_diseases.pkl')
