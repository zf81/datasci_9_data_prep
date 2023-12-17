import pandas as pd
import pickle
import joblib


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler



# Import the clean random sample of 10k data
df = pd.read_csv('model_dev1/data/processed/infectious_diseases_processed.csv')
len(df)


# drop rows with missing values
df.dropna(inplace=True)
len(df)

df.columns

# Define the features and the target variable
X = df.drop('sex', axis=1)  # Features (all columns except 'sex')
y = df['sex']               # Target variable (sex)

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X) # Fit the scaler to the features
pickle.dump(scaler, open('model_dev1/model/scaler_sex.sav', 'wb')) # Save the scaler for later use

# Fit the scaler to the features and transform
X_scaled = scaler.transform(X)

# Split the scaled data into training, validation, and testing sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Check the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev1/model/X_train_infectious.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev1/model/X_columns_infectious.sav', 'wb'))