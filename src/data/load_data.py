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
def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    # drop ids if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target to 0/1 if it's Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    # TotalCharges often has blanks in this dataset -> coerce to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # SeniorCitizen should be 0/1 ints if present
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)
        # Fill NA with median
        else:    
            df["SeniorCitizen"] = df["SeniorCitizen"].fillna(df["SeniorCitizen"].median()).astype(int)    
    
    return df
    # simple NA strategy:
    # - numeric: fill with 0    
    # - categorical: fill with mode
    # - boolean: fill with False
    # - ordinal: fill with mode
    # - datetime: fill with median 
    # - text: fill with mode
def encode_data(df: pd.DataFrame, target_col: str = "Churn", 
                categorical_cols: list = ["Gender", "SeniorCitizen", "Partner", "Dependents"], 
                numeric_cols: list = ["TotalCharges", "Age", "Seniority", "Education", "HoursPerWeek"], 
                ordinal_cols: list = ["MonthlyCharges"], 
                datetime_cols: list = ["LastCommunication", "LastContact", "LastAccountUpdate", "LastActivity"]) -> pd.DataFrame:
    """
    Encode categorical and numeric data.
    
    This function performs the following tasks:
    - Encode categorical data using one-hot encoding
    - Encode numeric data using ordinal encoding
    - Encode datetime data using ordinal encoding
    - Return the encoded data
    """
    # one-hot encode categorical data
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    df = onehot_encoder.fit_transform(df[categorical_cols])

    # ordinal encode numeric data
    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df = ordinal_encoder.fit_transform(df[numeric_cols])

    # ordinal encode datetime data
    datetime_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df = datetime_encoder.fit_transform(df[datetime_cols])

    return df
def create_features(df: pd.DataFrame, target_col: str = "Churn", 
                    categorical_cols: list = ["Gender", "SeniorCitizen", "Partner", "Dependents"], 
                    numeric_cols: list = ["TotalCharges", "Age", "Seniority", "Education", "HoursPerWeek"], 
                    ordinal_cols: list = ["MonthlyCharges"], 
                    datetime_cols: list = ["LastCommunication", "LastContact", "LastAccountUpdate", "LastActivity"]) -> pd.DataFrame:
    """ 
    Create new features from existing features.
    
    This function performs the following tasks:
    - Create new features from existing features
    - Return the new features
    """
    # create new features        
    new_features = df.copy()
    
    # calculate age in months
    new_features["AgeInMonths"] = new_features["Age"] / 12
    
    # calculate seniority in years
    new_features["SeniorityInYears"] = new_features["Seniority"] / 10
    
    # calculate seniority in months
    new_features["SeniorityInMonths"] = new_features["Seniority"] / 12
    
    # calculate seniority in days
    new_features["SeniorityInDays"] = new_features["Seniority"] / 365
    
    # calculate seniority in weeks
    new_features["SeniorityInWeeks"] = new_features["Seniority"] / 52    
    return new_features     
    # calculate hours per week in days
    new_features["HoursPerWeekInDays"] = new_features["HoursPerWeek"] / 24
    
    # calculate hours per week in weeks
    new_features["HoursPerWeekInWeeks"] = new_features["HoursPerWeek"] / 52
    
    # calculate hours per week in months    
    new_features["HoursPerWeekInMonths"] = new_features["HoursPerWeek"] / 4.33
    return new_features

def create_features(df: pd.DataFrame, target_col: str = "Churn", 
                    categorical_cols: list = ["Gender", "SeniorCitizen", "Partner", "Dependents"], 
                    numeric_cols: list = ["TotalCharges", "Age", "Seniority", "Education", "HoursPerWeek"], 
                    ordinal_cols: list = ["MonthlyCharges"], 
                    datetime_cols: list = ["LastCommunication", "LastContact", "LastAccountUpdate", "LastActivity"]) -> pd.DataFrame:
    """ 
    Create new features from existing features.
    
    This function performs the following tasks:
    - Create new features from existing features
    - Return the new features
    """     
    # create new features        
    new_features = df.copy()        
    # calculate age in months
    new_features["AgeInMonths"] = new_features["Age"] / 12
    
    # calculate seniority in years
    new_features["SeniorityInYears"] = new_features["Seniority"] / 10
    
    # calculate seniority in months
    new_features["SeniorityInMonths"] = new_features["Seniority"] / 12


    # calculate seniority in days
    new_features["SeniorityInDays"] = new_features["Seniority"] / 365
    
    # calculate seniority in weeks
    new_features["SeniorityInWeeks"] = new_features["Seniority"] / 52    
    return new_features     
    # calculate hours per week in days
    new_features["HoursPerWeekInDays"] = new_features["HoursPerWeek"] / 24
    
    # calculate hours per week in weeks
    new_features["HoursPerWeekInWeeks"] = new_features["HoursPerWeek"] / 52
     
    # calculate hours per week in months    
    new_features["HoursPerWeekInMonths"] = new_features["HoursPerWeek"] / 4.33
    return new_features
def _create_binary_mappings():
    """                         
    Create binary mappings for categorical features.            


    This function returns a dictionary mapping categorical feature values to binary values.
    """                         
    # create binary mappings for categorical features
    binary_mappings = {}
    for col in categorical_cols:
        binary_mappings[col] = {}
        for val in df[col].unique():
            binary_mappings[col][val] = 1 if val == "Yes" else 0
    return binary_mappings
def _create_ordinal_mappings():
    """                         
    Create ordinal mappings for numeric features.            


    This function returns a dictionary mapping numeric feature values to ordinal values.
    """                         
    # create ordinal mappings for numeric features
    ordinal_mappings = {}
    for col in numeric_cols:
        ordinal_mappings[col] = {}
        for val in df[col].unique():
            ordinal_mappings[col][val] = val
    return ordinal_mappings
def _create_datetime_mappings():
    """                         
    Create ordinal mappings for datetime features.            


    This function returns a dictionary mapping datetime feature values to ordinal values.
    """                         
    # create ordinal mappings for datetime features
    datetime_mappings = {}
    for col in datetime_cols:
        datetime_mappings[col] = {}
        for val in df[col].unique():
            datetime_mappings[col][val] = val           
    return datetime_mappings    
def _create_feature_transformer(categorical_cols, numeric_cols, ordinal_cols, datetime_cols):
    """                         
    Create a ColumnTransformer for new features.            


    This function returns a ColumnTransformer for new features.
    """                         
    # create a ColumnTransformer for new features
    feature_transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), numeric_cols),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
            ("datetime", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), datetime_cols)
        ]
    )
    return feature_transformer
    transformers = []

    # Encode categorical features
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore")))

        # Encode numeric features
    if numeric_cols:
        transformers.append(("numeric", OrdinalEncoder(handle_unknown="ignore")))
    # Encode ordinal features
    if ordinal_cols:
        transformers.append(("ordinal", OrdinalEncoder(handle_unknown="ignore")))
    # Encode datetime features
    if datetime_cols:
        transformers.append(("datetime", OrdinalEncoder(handle_unknown="ignore")))
    # Create the ColumnTransformer
    feature_transformer = ColumnTransformer(transformers=transformers, remainder="passthrough")

    return feature_transformer
def _create_feature_transformer(categorical_cols, numeric_cols, ordinal_cols, datetime_cols):
    """ 
    Create a feature transformer that will be used to encode categorical and numeric features.
    """
    # Create a list of transformers to be used in the feature transformer
    transformers = []

    # Encode categorical features
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore")))

    # Encode numeric features
    if numeric_cols:
        transformers.append(("numeric", OrdinalEncoder(handle_unknown="ignore")))


    # Encode ordinal features
    if ordinal_cols:
        transformers.append(("ordinal", OrdinalEncoder(handle_unknown="ignore")))

    # Encode datetime features
    if datetime_cols:
        transformers.append(("datetime", OrdinalEncoder(handle_unknown="ignore")))

    # Create the ColumnTransformer
    feature_transformer = ColumnTransformer(transformers=transformers, remainder="passthrough")

    return feature_transformer
    # create a ColumnTransformer for new features
    feature_transformer = ColumnTransformer(        
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", OrdinalEncoder(handle_unknown="ignore"), numeric_cols),
            ("ordinal", OrdinalEncoder(handle_unknown="ignore"), ordinal_cols),
            ("datetime", OrdinalEncoder(handle_unknown="ignore"), datetime_cols),
        ],
        remainder="passthrough",
    )
    return feature_transformer

    return feature_transformer
    return feature_transformer
def build_features(df, categorical_cols, numeric_cols, ordinal_cols, datetime_cols):
    """ 
    Build features from categorical and numeric features.
    """
    # Create a list of transformers to be used in the feature transformer
    transformers = []

    # Encode categorical features
    if categorical_cols:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore")))

    # z
    if numeric_cols:
        transformers.append(("numeric", OrdinalEncoder(handle_unknown="ignore")))

