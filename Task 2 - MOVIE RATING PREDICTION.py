import pandas as pd
from sklearn.impute import SimpleImputer

# Function to load and display dataset information
def load_and_display_dataset(file_path):
    dataset = pd.read_csv(file_path)
    rows, columns = dataset.shape
    print(f'The dataset has {rows} rows and {columns} columns.')
    return dataset

# Function to summarize missing data
def summarize_missing_data(df):
    missing_data_summary = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_data_summary)
    return missing_data_summary

# Function to handle missing values
def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            imputer = SimpleImputer(strategy='most_frequent')
            df[column] = imputer.fit_transform(df[[column]])
        else:
            imputer = SimpleImputer(strategy='mean')
            df[column] = imputer.fit_transform(df[[column]])
    return df

# Function to convert data types
def convert_data_types(df):
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0).astype(int)
    return df

# Function to analyze target variable distribution
def analyze_target_distribution(df, target_column):
    target_distribution = df[target_column].value_counts()
    print(f"Distribution of '{target_column}':")
    print(target_distribution)
    return target_distribution

# Function to display the first few rows of the dataset
def display_first_rows(df, num_rows=5):
    print(df.head(num_rows))

# File path to the dataset
file_path = 'path_to_your_dataset.csv'

# Load and display dataset information
dataset = load_and_display_dataset(file_path)

# Summarize missing data
summarize_missing_data(dataset)

# Handle missing values
dataset = handle_missing_values(dataset)

# Convert data types
dataset = convert_data_types(dataset)

# Analyze the distribution of the target variable
target_column = 'Aggregate rating'
analyze_target_distribution(dataset, target_column)

# Display the first few rows after preprocessing
display_first_rows(dataset)
