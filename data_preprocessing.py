import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Handles loading and initial processing of NASA CMAPSS dataset.
    The dataset has no column names, so we add them manually.
    """
    
    def __init__(self):
        # Define column names for the dataset
        # NASA CMAPSS has: unit_id, time_cycles, 3 operational settings, 21 sensor readings
        self.index_names = ['unit_id', 'time_cycles']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names
    
    def load_data(self, filepath, has_rul=False):
        """
        Load the NASA CMAPSS dataset from text file.
        
        Args:
            filepath: Path to the train or test file
            has_rul: True for train data (we calculate RUL), False for test data
        
        Returns:
            DataFrame with proper column names
        """
        # Read space-separated file without headers
        df = pd.read_csv(filepath, sep='\s+', header=None, names=self.col_names)
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Number of engines: {df['unit_id'].nunique()}")
        
        return df
    
    def add_rul(self, df):
        """
        Calculate Remaining Useful Life (RUL) for training data.
        
        RUL Logic:
        - For each engine, find the maximum cycle (when it failed)
        - RUL at any cycle = max_cycle - current_cycle
        - At cycle 1, RUL is highest; at last cycle, RUL = 0
        
        Args:
            df: Training dataframe
            
        Returns:
            DataFrame with RUL column added
        """
        # Group by engine and get max cycles for each
        max_cycles = df.groupby('unit_id')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']
        
        # Merge max cycles back to original data
        df = df.merge(max_cycles, on='unit_id', how='left')
        
        # Calculate RUL: remaining cycles until failure
        df['RUL'] = df['max_cycle'] - df['time_cycles']
        
        # Drop the helper column
        df.drop('max_cycle', axis=1, inplace=True)
        
        print(f"RUL range: {df['RUL'].min()} to {df['RUL'].max()}")
        
        return df


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    
    # Load training data
    train_df = loader.load_data('train_FD001.txt')
    train_df = loader.add_rul(train_df)
    
    print("\nSample data:")
    print(train_df.head(10))
    print("\nData info:")
    print(train_df.info())