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