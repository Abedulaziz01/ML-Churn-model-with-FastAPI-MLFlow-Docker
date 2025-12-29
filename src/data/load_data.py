
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path: str, test_size: float = 0.2, random_state: int = 42):    
    """
    Load and preprocess data from a CSV file.
    
    This function performs the following tasks:
    - Reads the CSV file into a Pandas DataFrame
    - Preprocesses the data using the preprocess_data function
    - Splits the data into training and testing sets
    - Returns the training and testing sets
    """
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_col]), 
                                                       df[target_col], test_size=test_size, 
                                                       random_state=random_state)
    
    return X_train, X_test, y_train, y_test
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Preprocess the data by encoding categorical variables and handling missing values.
    
    This function performs the following tasks:
    - Encodes categorical variables using OrdinalEncoder
    - Handles missing values using the fillna() method
    - Returns the preprocessed DataFrame
    """
    # Encode categorical variables using OrdinalEncoder
    encoder = OrdinalEncoder()
    df = encoder.fit_transform(df)

    # Handle missing values using the fillna() method
    df = df.fillna(0)
    
    return df
     

# Define the target column
target_col = 'target'  # Replace 'target' with the actual target column name in your dataset    

# Load the data from the CSV file
X_train, X_test, y_train, y_test = load_data('data/train.csv', test_size=0.2, random_state=42)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Define the target column
target_col = 'target'  # Replace 'target' with the actual target column name in your dataset

# Define the features columns
features_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']

# Define the target column
target_col = 'target'  # Replace 'target' with the actual target column name in your dataset

# Define the features columns
features_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
categorical_cols = ['feature1', 'feature2']  # Replace with actual categorical feature names
numerical_cols = ['feature3', 'feature4', 'feature5']  # Replace with actual numerical feature names
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)  
# Define the features columns
features_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
categorical_cols = ['feature1', 'feature2']  # Replace with actual categorical feature names
numerical_cols = ['feature3', 'feature4', 'feature5']  # Replace with actual numerical feature names
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
    