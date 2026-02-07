import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

class DataPreparation:
    """
    Prepares data for model training by scaling features and creating sequences.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()  # Scales features to [0, 1] range
        self.feature_columns = None
    
    def prepare_for_ml(self, df, is_train=True):
        """
        Prepare data for machine learning models (non-sequential).
        
        Args:
            df: Input dataframe with features
            is_train: True for training data (fit scaler), False for test (use fitted scaler)
            
        Returns:
            X (features), y (RUL target if available)
        """
        # Identify feature columns (exclude identifiers and target)
        exclude_cols = ['unit_id', 'time_cycles', 'RUL']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Number of features: {len(self.feature_columns)}")
        
        # Extract features
        X = df[self.feature_columns].values
        
        # Scale features
        if is_train:
            X_scaled = self.scaler.fit_transform(X)
            print("Fitted scaler on training data")
        else:
            X_scaled = self.scaler.transform(X)
            print("Applied existing scaler to data")
        
        # Extract target if available (training data)
        y = df['RUL'].values if 'RUL' in df.columns else None
        
        return X_scaled, y
    
    def create_sequences(self, df, sequence_length=30):
        """
        Create sequences for LSTM/RNN models.
        
        Why sequences: LSTMs need sequences of data points to learn temporal patterns.
        Each sequence represents the last N cycles of an engine's operation.
        
        Args:
            df: Input dataframe
            sequence_length: Number of time steps in each sequence
            
        Returns:
            X_seq: 3D array (samples, time_steps, features)
            y_seq: Target RUL values
            engine_ids: Corresponding engine IDs for each sequence
        """
        exclude_cols = ['unit_id', 'time_cycles', 'RUL']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_sequences = []
        y_sequences = []
        engine_ids = []
        
        # Process each engine separately
        for engine_id in df['unit_id'].unique():
            engine_data = df[df['unit_id'] == engine_id]
            
            # Get feature values and RUL
            features = engine_data[feature_cols].values
            rul_values = engine_data['RUL'].values if 'RUL' in df.columns else None
            
            # Create sequences with sliding window
            for i in range(sequence_length, len(engine_data) + 1):
                # Take last 'sequence_length' cycles as one sequence
                X_sequences.append(features[i-sequence_length:i])
                
                if rul_values is not None:
                    # Target is the RUL at the last time step of the sequence
                    y_sequences.append(rul_values[i-1])
                
                engine_ids.append(engine_id)
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences) if y_sequences else None
        
        print(f"Created sequences: X shape = {X_seq.shape}, y shape = {y_seq.shape if y_seq is not None else 'None'}")
        
        return X_seq, y_seq, engine_ids
    
    def prepare_test_data(self, test_df, rul_df=None):
        """
        Prepare test data - we only use the last sequence for each engine.
        
        Why: For testing, we want to predict the RUL at the last known cycle
        of each engine (simulating real-world scenario where we predict from
        current state).
        
        Args:
            test_df: Test dataframe
            rul_df: DataFrame with actual RUL values (if available)
            
        Returns:
            DataFrame with last cycle of each engine
        """
        # Get the last cycle for each engine
        last_cycles = test_df.groupby('unit_id').last().reset_index()
        
        # Add actual RUL if provided
        if rul_df is not None:
            last_cycles = last_cycles.merge(rul_df, on='unit_id', how='left')
        
        print(f"Prepared test data: {len(last_cycles)} engines")
        
        return last_cycles
    
    def save_scaler(self, filepath='scaler.pkl'):
        """
        Save the fitted scaler for later use.
        
        Why: We need the same scaling transformation for test data
        that was used on training data.
        """
        # Save both the scaler and the feature column ordering so we can
        # apply the same transformation later during prediction.
        payload = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        print(f"Scaler and feature columns saved to {filepath}")
    
    def load_scaler(self, filepath='scaler.pkl'):
        """
        Load a previously saved scaler.
        """
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)

        # Support both legacy scaler-only files and the new payload format
        if isinstance(payload, dict) and 'scaler' in payload:
            self.scaler = payload['scaler']
            self.feature_columns = payload.get('feature_columns')
            print(f"Scaler and feature columns loaded from {filepath}")
        else:
            # Legacy: payload is the scaler object
            self.scaler = payload
            print(f"Scaler loaded from {filepath} (no feature columns found)")
    
    def clip_rul(self, rul_values, max_rul=125):
        """
        Clip RUL values to a maximum threshold.
        
        Why: Very early in an engine's life, the RUL can be 200+ cycles.
        But predicting "exactly 200 vs 180" isn't useful. We care more about
        accuracy when failure is near. Clipping helps the model focus on 
        the critical range.
        
        Args:
            rul_values: Array of RUL values
            max_rul: Maximum RUL threshold
            
        Returns:
            Clipped RUL values
        """
        return np.clip(rul_values, 0, max_rul)


# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataLoader
    from feature_engineering import FeatureEngineer
    
    # Load and engineer features
    loader = DataLoader()
    train_df = loader.load_data('train_FD001.txt')
    train_df = loader.add_rul(train_df)
    
    engineer = FeatureEngineer()
    train_df = engineer.engineer_all_features(train_df, simple=True)
    
    # Prepare data
    prep = DataPreparation()
    X_train, y_train = prep.prepare_for_ml(train_df, is_train=True)
    
    # Clip RUL for better training
    y_train_clipped = prep.clip_rul(y_train, max_rul=125)
    
    print(f"\nPrepared data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_train range: {y_train.min()} to {y_train.max()}")
    print(f"y_train_clipped range: {y_train_clipped.min()} to {y_train_clipped.max()}")
    
    # Save scaler
    prep.save_scaler()