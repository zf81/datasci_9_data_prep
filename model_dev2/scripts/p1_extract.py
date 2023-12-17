import pandas as pd 

# This dataset includes all valid felony, misdemeanor, and violation crimes reported to the New York City Police Department (NYPD) 
# for all complete quarters so far this year

datalink = 'https://data.cityofnewyork.us/api/views/5uac-w243/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.size
df.sample(5)

## save as csv to model_dev2/data/raw/
df.to_csv('model_dev2/data/raw/complaint.csv', index=False)

## save as pickle to model_dev2/data/raw/
df.to_pickle('model_dev2/data/raw/complaint.pkl')
