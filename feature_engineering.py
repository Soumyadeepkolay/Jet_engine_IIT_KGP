import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Creates additional features from raw sensor data to improve model performance.
    Feature engineering is crucial for predictive maintenance.
    """
    
    def __init__(self):
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    
    def create_rolling_features(self, df, window_sizes=[5, 10, 15]):
        """
        Create rolling mean and std features for sensors.
        
        Why: Rolling statistics capture trends over time, which helps detect 
        gradual degradation patterns in engine health.
        
        Args:
            df: Input dataframe
            window_sizes: List of window sizes for rolling calculations
            
        Returns:
            DataFrame with rolling features added
        """
        df_new = df.copy()
        
        for window in window_sizes:
            for sensor in self.sensor_cols:
                # Rolling mean captures average trend
                df_new[f'{sensor}_rolling_mean_{window}'] = df_new.groupby('unit_id')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std captures variability/stability
                df_new[f'{sensor}_rolling_std_{window}'] = df_new.groupby('unit_id')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        # Fill NaN values from std calculation with 0
        df_new.fillna(0, inplace=True)
        
        print(f"Added rolling features. New shape: {df_new.shape}")
        return df_new
    
    def create_lag_features(self, df, lag_steps=[1, 2, 3]):
        """
        Create lag features (previous values) for sensors.
        
        Why: Comparing current values with past values helps identify 
        acceleration of degradation.
        
        Args:
            df: Input dataframe
            lag_steps: List of lag steps
            
        Returns:
            DataFrame with lag features
        """
        df_new = df.copy()
        
        for lag in lag_steps:
            for sensor in self.sensor_cols:
                df_new[f'{sensor}_lag_{lag}'] = df_new.groupby('unit_id')[sensor].shift(lag)
        
        # Fill NaN with forward fill (use current value for missing lags)
        df_new.fillna(method='bfill', inplace=True)
        
        print(f"Added lag features. New shape: {df_new.shape}")
        return df_new
    
    def create_rate_of_change(self, df):
        """
        Calculate rate of change (difference from previous cycle) for sensors.
        
        Why: Rate of change indicates how fast a sensor value is deteriorating,
        which is a strong signal for upcoming failure.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with rate of change features
        """
        df_new = df.copy()
        
        for sensor in self.sensor_cols:
            # Calculate difference from previous cycle
            df_new[f'{sensor}_roc'] = df_new.groupby('unit_id')[sensor].diff()
        
        # Fill first value of each engine with 0
        df_new.fillna(0, inplace=True)
        
        print(f"Added rate of change features. New shape: {df_new.shape}")
        return df_new
    
    def create_statistical_features(self, df):
        """
        Create statistical aggregations for each engine up to current cycle.
        
        Why: Overall statistics (like mean, max, min) provide context about
        the engine's historical behavior patterns.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with statistical features
        """
        df_new = df.copy()
        
        for sensor in self.sensor_cols:
            # Cumulative mean up to current cycle
            df_new[f'{sensor}_cum_mean'] = df_new.groupby('unit_id')[sensor].transform(
                lambda x: x.expanding().mean()
            )
            
            # Cumulative max
            df_new[f'{sensor}_cum_max'] = df_new.groupby('unit_id')[sensor].transform(
                lambda x: x.expanding().max()
            )
            
            # Cumulative min
            df_new[f'{sensor}_cum_min'] = df_new.groupby('unit_id')[sensor].transform(
                lambda x: x.expanding().min()
            )
        
        print(f"Added statistical features. New shape: {df_new.shape}")
        return df_new
    
    def remove_constant_features(self, df):
        """
        Remove features that have zero variance (constant values).
        
        Why: Constant features provide no information for prediction and 
        only increase computation.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with constant features removed
        """
        # Identify numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate variance for each column
        variances = df[numeric_cols].var()
        
        # Find columns with zero or near-zero variance
        constant_cols = variances[variances < 0.001].index.tolist()
        
        print(f"Removing {len(constant_cols)} constant features: {constant_cols}")
        
        # Drop constant columns
        df_new = df.drop(columns=constant_cols)
        
        return df_new
    
    def engineer_all_features(self, df, simple=False):
        """
        Apply all feature engineering steps.
        
        Args:
            df: Input dataframe
            simple: If True, only use basic features (faster, less memory)
            
        Returns:
            Fully engineered DataFrame
        """
        print("Starting feature engineering...")
        
        if simple:
            # Minimal feature engineering for faster training
            df = self.create_rolling_features(df, window_sizes=[5])
            df = self.create_rate_of_change(df)
        else:
            # Full feature engineering for better accuracy
            df = self.create_rolling_features(df, window_sizes=[5, 10, 15])
            df = self.create_lag_features(df, lag_steps=[1, 2, 3])
            df = self.create_rate_of_change(df)
            df = self.create_statistical_features(df)
        
        df = self.remove_constant_features(df)
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df


# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataLoader
    
    loader = DataLoader()
    train_df = loader.load_data('train_FD001.txt')
    train_df = loader.add_rul(train_df)
    
    engineer = FeatureEngineer()
    train_df_engineered = engineer.engineer_all_features(train_df, simple=True)
    
    print("\nEngineered features sample:")
    print(train_df_engineered.head())