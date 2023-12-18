# datasci_9_data_prep

Selecting datasets suitable for a machine learning experiment, with an emphasis on data cleaning, encoding, and transformation steps necessary to prepare the data.

Dataset Selection: 

The first dataset was found on [Data.gov] (https://data.gov/): [Infectious Diseases by Disease, County, Year, and Sex] (https://catalog.data.gov/dataset/infectious-diseases-by-disease-county-year-and-sex-d8912). It contains case counts and rates for selected communicable diseases. It is reported for California residents, by disease, county, year, and sex. The data represent cases with an estimated illness onset date from 2001 through the last year indicated from California Confidential Morbidity Reports and/or Laboratory Reports. The inteded machine learning task for this dataset was classification. 

The second dataset was found on [Data.gov] (https://data.gov/): [NYPD Complaint Data Current (Year To Date)] (https://catalog.data.gov/dataset/nypd-complaint-data-current-year-to-date). It contains all valid felony, misdemeanor, and violation crimes reported to the New York City Police Department (NYPD) for all complete quarters so far this year (2023). The inteded machine learning task for this dataset was regression. 

Data Cleaning and Transformation:

For sklearn to load in Visual Studio Code, I had to first install it on my computer using, pip install sklearn in the terminal

Setting up folder and file structure: 
- Create a model_dev1 folder. All sub-folders and files within this folder will be used for the Infectious Disease dataset
- In model_dev1, create 3 folders with the following names: data, model, and scripts
- In the data folder, create two more folders with the following names: processed and raw
- In the scripts folder, create three python (.py) files with the following names: p1_extract.py, p2_transform.py, and p3_compute.py
- In p1_extract.py file, load in the Infectious Disease dataset and save it as both a .csv file and a .pkl file to the raw folder
- In p2_transform.py file, load in the .pkl file from the raw folder
- For the NYPD Complaints dataset, create a new folder called model_dev2 and repeat steps for creating sub-folders and files as above

Clean data: 
- To clean column names, removing white spaces, special characters, and make all letters lowercase
- Examine the data types and make sure they match each column. Be sure to convert categorical columns into objects
- Drop rows that contain missing values 
- Select columns to keep and columns to drop based on what will be used for the machine learning task
- For the cleaned dataset, check that it contains the columns you intend to keep






