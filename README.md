# datasci_9_data_prep

Selecting datasets suitable for a machine learning experiment, with an emphasis on data cleaning, encoding, and transformation steps necessary to prepare the data.

# Dataset Selection: 

The first dataset was found on [Data.gov](https://data.gov/): [Infectious Diseases by Disease, County, Year, and Sex](https://catalog.data.gov/dataset/infectious-diseases-by-disease-county-year-and-sex-d8912). It contains case counts and rates for selected communicable diseases. It is reported for California residents, by disease, county, year, and sex. The data represent cases with an estimated illness onset date from 2001 through the last year indicated from California Confidential Morbidity Reports and/or Laboratory Reports. The inteded machine learning task for this dataset was classification. 

The second dataset was found on [Data.gov](https://data.gov/): [NYPD Complaint Data Current (Year To Date)](https://catalog.data.gov/dataset/nypd-complaint-data-current-year-to-date). It contains all valid felony, misdemeanor, and violation crimes reported to the New York City Police Department (NYPD) for all complete quarters so far this year (2023). The inteded machine learning task for this dataset was regression. 

# Data Cleaning and Transformation:

For sklearn to load in Visual Studio Code, I had to first install it on my computer using, <code>pip install sklearn</code> in the terminal

**Setting up folder and file structure:**
- Create a model_dev1 folder. All sub-folders and files within this folder will be used for the Infectious Disease dataset
- In model_dev1, create 3 folders with the following names: <code>data</code>, <code>model</code>, and <code>scripts</code> 
- In the <code>data</code> folder, create two more folders with the following names: <code>processed</code> and <code>raw</code> 
- In the <code>scripts</code> folder, create three python (.py) files with the following names: <code>p1_extract.py</code>, <code>p2_transform.py</code>, and <code>p3_compute.py</code>
- In <code>p1_extract.py</code> file, load in the Infectious Disease dataset and save it as both a .csv file and a .pkl file to the raw folder
- In <code>p2_transform.py</code> file, load in the .pkl file from the <code>raw</code> folder
- For the NYPD Complaints dataset, create a new folder called <code>model_dev2</code> and repeat steps for creating sub-folders and files as above

# Cleaning Data: 
- To clean column names, remove white spaces, special characters, and make all letters lowercase
- Examine the data types and make sure they match each column. Be sure to convert categorical columns into objects
- Drop rows that contain missing values 
- Select columns to keep and columns to drop based on what will be used for the machine learning task
- For the cleaned dataset, check that it contains the columns you intend to keep

# Transforming Data:
- Perform ordinal encoding on selected columns with categorical values
- Create a dataset for the encoding with mapping
- Save this mapping dataset as a .csv file to the <code>processed</code> data folder
- Repeat the above steps for additional columns as needed
- Save a temporary .csv file of the encoded dataset to the <code>processed</code> data folder 
- Check that all values in the new dataset are numerical

# Dataset Splitting: 
- In the <code>p3_compute.py</code> file, load in the processed dataset
- Define the independent and dependent (target) variables
- The independent variables/features are the columns other than the target/dependent variable
- Use the code for standard scaler and fit it to the features
- Save the scaler as a .pkl file to the <code>model</code> folder
- Fit the scaler to the feauers and transform
- Split the scaled data into training, validation, and testing sets
- Save the X_train and X.columns to the model folder
- Create a baseline model using DummyClassifier
- Using the train and value variables from the data splitting step, create logistic regression models

# Errors:
- When pushing the code from Visual Studio Code to my github repository, I received a few errors including "failed to push some refs" and "pre-receive hook declined." The NYPD dataset was larger than 25mb, and so I listed that dataset in a .env file and placed the .env file to a .gitignore file. I removed the larger dataset from my VSCode workspace and I also re-checked the size of each file under all of the folders in model_dev1 and model_dev2. However, none of my changes were being pushed to the github repository. I deleted and created a new repository and opened all the folders and files for this assignment in a new VSCode window. I was finally able to push and sync all changes to the repository for this assignment. 






