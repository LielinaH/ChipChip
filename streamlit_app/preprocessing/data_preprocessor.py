import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, engine):
        self.engine = engine
        self.label_encoders = {}

    def load_data(self, table_name):
        """
        Load data from the database.
        
        Args:
            table_name (str): Name of the table to load data from.
        
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, self.engine)
        return df

    def handle_nulls(self, df, strategy='mean'):
        """
        Handle NULL values using the specified strategy.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            strategy (str): Strategy to handle NULLs ('mean', 'median', 'mode', 'constant').
        
        Returns:
            pd.DataFrame: DataFrame with NULL values handled.
        """
        if strategy in ['mean', 'median', 'mode']:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if strategy == 'mean':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == 'median':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif strategy == 'mode':
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
        elif strategy == 'constant':
            df = df.fillna(0)
        else:
            raise ValueError("Invalid null strategy. Use 'mean', 'median', 'mode', or 'constant'.")
        return df

    def encode_categorical(self, df, columns, encoding_type='onehot'):
        """
        Encode categorical variables using the specified encoding type.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of categorical columns to encode.
            encoding_type (str): Encoding type ('onehot' or 'label').
        
        Returns:
            pd.DataFrame: DataFrame with categorical variables encoded.
        """
        if encoding_type == 'onehot':
            return pd.get_dummies(df, columns=columns)
        elif encoding_type == 'label':
            for column in columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    df[column] = self.label_encoders[column].fit_transform(df[column])
                else:
                    df[column] = self.label_encoders[column].transform(df[column])
            return df
        else:
            raise ValueError("Invalid encoding type. Use 'onehot' or 'label'.")

    def aggregate_timestamps(self, df, timestamp_column, period='M'):
        """
        Aggregate timestamps to specified periods.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            timestamp_column (str): Column containing timestamp values.
            period (str): Resampling period ('D', 'W', 'M', 'Y').
        
        Returns:
            pd.DataFrame: DataFrame with aggregated timestamps.
        """
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df[f'{timestamp_column}_period'] = df[timestamp_column].dt.to_period(period)
        return df

    def preprocess(self, table_name, null_strategy='mean', categorical_columns=None, encoding_type='onehot', timestamp_column=None, period='M'):
        """
        Preprocess and normalize data for a specific table.
        
        Parameters:
            table_name (str): Name of the database table to preprocess.
            null_strategy (str): Strategy to handle NULLs ('mean', 'median', 'mode', 'constant').
            categorical_columns (list): List of categorical columns to encode.
            encoding_type (str): Encoding type for categorical variables ('onehot' or 'label').
            timestamp_column (str): Column containing timestamp values.
            period (str): Resampling period for timestamps ('D', 'W', 'M', 'Y').
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        data = self.load_data(table_name)

        # Handle NULL values
        data = self.handle_nulls(data, strategy=null_strategy)

        # Encode categorical variables
        if categorical_columns:
            data = self.encode_categorical(data, columns=categorical_columns, encoding_type=encoding_type)

        # Aggregate timestamps
        if timestamp_column:
            data = self.aggregate_timestamps(data, timestamp_column=timestamp_column, period=period)

        return data