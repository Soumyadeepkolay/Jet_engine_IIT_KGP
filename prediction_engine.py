import numpy as np
import pandas as pd
import pickle
import os

class PredictionEngine:
    """
    Complete end-to-end pipeline for making predictions on new engine data.
    This ties together all the components: loading, preprocessing, prediction, health scoring.
    """
    
    def __init__(self, model_path='rul_model.pkl', scaler_path='scaler.pkl'):
        """
        Initialize prediction engine with trained model and scaler.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved scaler
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Initialize components
        from data_preprocessing import DataLoader
        from feature_engineering import FeatureEngineer
        from health_score_calculator import HealthScoreCalculator
        
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.health_calculator = HealthScoreCalculator(max_rul=125)
    
    def load_artifacts(self):
        """
        Load the trained model and scaler from disk.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_path}")
        
        # Load scaler (supports payload with scaler + feature_columns)
        with open(self.scaler_path, 'rb') as f:
            payload = pickle.load(f)

        if isinstance(payload, dict) and 'scaler' in payload:
            self.scaler = payload['scaler']
            # If feature columns were saved with the scaler, use them to
            # ensure consistent feature ordering at prediction time.
            self.feature_columns = payload.get('feature_columns')
            print(f"Scaler and feature columns loaded from {self.scaler_path}")
        else:
            # Legacy: scaler-only file
            self.scaler = payload
            print(f"Scaler loaded from {self.scaler_path} (no feature columns found)")
    
    def preprocess_data(self, df, simple_features=True):
        """
        Apply the complete preprocessing pipeline.
        
        Args:
            df: Raw dataframe from test file
            simple_features: Use simple feature engineering (faster)
            
        Returns:
            Preprocessed dataframe ready for prediction
        """
        print("Preprocessing data...")
        
        # Apply feature engineering
        df_engineered = self.feature_engineer.engineer_all_features(df, simple=simple_features)
        
        return df_engineered
    
    def predict_single_engine(self, engine_id, test_df):
        """
        Make prediction for a specific engine.
        
        Args:
            engine_id: ID of the engine to predict
            test_df: Preprocessed test dataframe
            
        Returns:
            Health report dictionary
        """
        # Filter data for this engine
        engine_data = test_df[test_df['unit_id'] == engine_id]
        
        if len(engine_data) == 0:
            raise ValueError(f"Engine ID {engine_id} not found in data")
        
        # Get the last cycle (current state)
        last_cycle = engine_data.iloc[-1:].copy()

        # Determine feature ordering: prefer saved training ordering
        exclude_cols = ['unit_id', 'time_cycles', 'RUL']
        if self.feature_columns is None:
            feature_cols = [col for col in last_cycle.columns if col not in exclude_cols]
            self.feature_columns = feature_cols
        else:
            feature_cols = self.feature_columns

        # Reindex to match training features; fill missing features with 0
        features_df = last_cycle.reindex(columns=feature_cols).fillna(0)
        X = features_df.values

        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        rul_prediction = self.model.predict(X_scaled)[0]
        rul_prediction = max(0, rul_prediction)  # Ensure non-negative
        
        # Generate health report
        report = self.health_calculator.create_health_report(engine_id, rul_prediction)
        
        return report
    
    def predict_all_engines(self, test_df):
        """
        Make predictions for all engines in the test set.
        
        Args:
            test_df: Preprocessed test dataframe
            
        Returns:
            DataFrame with predictions and health scores for all engines
        """
        print("Making predictions for all engines...")
        
        # Get last cycle for each engine
        last_cycles = test_df.groupby('unit_id').last().reset_index()
        
        # Determine feature columns to use (prefer saved training order)
        exclude_cols = ['unit_id', 'time_cycles', 'RUL']
        if self.feature_columns is None:
            # Fallback: infer from test data
            feature_cols = [col for col in last_cycles.columns if col not in exclude_cols]
            self.feature_columns = feature_cols
        else:
            feature_cols = self.feature_columns

        # Reindex test data to match training feature order. Missing columns
        # (present during training but not in test) are filled with 0. Extra
        # columns present in test but not used in training are ignored.
        features_df = last_cycles.reindex(columns=feature_cols).fillna(0)
        X = features_df.values
        engine_ids = last_cycles['unit_id'].values
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        rul_predictions = self.model.predict(X_scaled)
        rul_predictions = np.maximum(rul_predictions, 0)  # Ensure non-negative
        
        # Generate health scores
        health_df = self.health_calculator.batch_health_scores(engine_ids, rul_predictions)
        
        print(f"Predictions complete for {len(health_df)} engines")
        
        return health_df
    
    def predict_engine_trajectory(self, engine_id, test_df, num_future_cycles=50):
        """
        Predict health trajectory for an engine over multiple future cycles.
        
        This creates the health degradation graph over time.
        
        Args:
            engine_id: Engine to predict
            test_df: Test dataframe
            num_future_cycles: How many cycles ahead to project
            
        Returns:
            DataFrame with cycle-by-cycle predictions
        """
        # Get engine data
        engine_data = test_df[test_df['unit_id'] == engine_id].copy()
        
        if len(engine_data) == 0:
            raise ValueError(f"Engine ID {engine_id} not found")
        
        # Prepare feature extraction
        exclude_cols = ['unit_id', 'time_cycles', 'RUL']
        if self.feature_columns is None:
            feature_cols = [col for col in engine_data.columns if col not in exclude_cols]
            self.feature_columns = feature_cols
        else:
            feature_cols = self.feature_columns

        # Reindex engine data to match training features and fill missing with 0
        features_all_df = engine_data.reindex(columns=feature_cols).fillna(0)

        # Get predictions for each existing cycle
        X_all = features_all_df.values
        X_scaled = self.scaler.transform(X_all)
        rul_predictions = self.model.predict(X_scaled)
        rul_predictions = np.maximum(rul_predictions, 0)
        
        # Calculate health percentages
        health_percentages = [
            self.health_calculator.rul_to_health_percentage(rul) 
            for rul in rul_predictions
        ]
        
        # Create trajectory dataframe
        trajectory = pd.DataFrame({
            'cycle': engine_data['time_cycles'].values,
            'rul': rul_predictions,
            'health_percentage': health_percentages,
            'status': [
                self.health_calculator.get_health_status(hp) 
                for hp in health_percentages
            ]
        })
        
        # Project future cycles (simple linear extrapolation)
        if num_future_cycles > 0:
            last_cycle = trajectory['cycle'].iloc[-1]
            last_rul = trajectory['rul'].iloc[-1]
            
            # Estimate degradation rate
            if len(trajectory) > 1:
                rul_change = trajectory['rul'].iloc[-1] - trajectory['rul'].iloc[0]
                cycles_elapsed = trajectory['cycle'].iloc[-1] - trajectory['cycle'].iloc[0]
                degradation_rate = rul_change / cycles_elapsed if cycles_elapsed > 0 else -1
            else:
                degradation_rate = -1  # Default: 1 RUL per cycle
            
            # Project future
            future_cycles = []
            for i in range(1, num_future_cycles + 1):
                future_cycle = last_cycle + i
                future_rul = max(0, last_rul + (degradation_rate * i))
                future_health = self.health_calculator.rul_to_health_percentage(future_rul)
                future_status = self.health_calculator.get_health_status(future_health)
                
                future_cycles.append({
                    'cycle': future_cycle,
                    'rul': future_rul,
                    'health_percentage': future_health,
                    'status': future_status,
                    'projected': True
                })
            
            # Add projected flag to existing data
            trajectory['projected'] = False
            
            # Combine
            future_df = pd.DataFrame(future_cycles)
            trajectory = pd.concat([trajectory, future_df], ignore_index=True)
        
        return trajectory
    
    def generate_health_graph_data(self, engine_id, test_df):
        """
        Generate data specifically for the health graph visualization.
        This is what judges will see when they pick a random engine.
        
        Args:
            engine_id: Engine ID to visualize
            test_df: Test dataframe
            
        Returns:
            Dictionary with graph data and metadata
        """
        # Get trajectory
        trajectory = self.predict_engine_trajectory(engine_id, test_df, num_future_cycles=30)
        
        # Get current health report
        current_report = self.predict_single_engine(engine_id, test_df)
        
        # Prepare graph data
        graph_data = {
            'engine_id': engine_id,
            'current_health': current_report['health_percentage'],
            'current_rul': current_report['rul_cycles'],
            'current_status': current_report['status'],
            'recommendation': current_report['recommendation'],
            'trajectory': trajectory.to_dict('records'),
            'cycles': trajectory['cycle'].tolist(),
            'health_values': trajectory['health_percentage'].tolist(),
            'status_values': trajectory['status'].tolist()
        }
        
        return graph_data
    
    def save_predictions(self, health_df, filepath='predictions.csv'):
        """
        Save predictions to CSV file.
        
        Args:
            health_df: DataFrame with predictions
            filepath: Output file path
        """
        health_df.to_csv(filepath, index=False)
        print(f"Predictions saved to {filepath}")


def run_complete_pipeline(train_file, test_file, simple_features=True):
    """
    Run the complete training and prediction pipeline.
    
    This function demonstrates the entire workflow from data loading to predictions.
    
    Args:
        train_file: Path to training data
        test_file: Path to test data
        simple_features: Use simple feature engineering
        
    Returns:
        Prediction engine and results
    """
    from data_preprocessing import DataLoader
    from feature_engineering import FeatureEngineer
    from data_preparation import DataPreparation
    from model_training import RULModel
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("STEP 1: Loading Training Data")
    print("=" * 60)
    loader = DataLoader()
    train_df = loader.load_data(train_file)
    train_df = loader.add_rul(train_df)
    
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering on Training Data")
    print("=" * 60)
    engineer = FeatureEngineer()
    train_df = engineer.engineer_all_features(train_df, simple=simple_features)
    
    print("\n" + "=" * 60)
    print("STEP 3: Preparing Data for Training")
    print("=" * 60)
    prep = DataPreparation()
    X, y = prep.prepare_for_ml(train_df, is_train=True)
    y = prep.clip_rul(y, max_rul=125)
    
    # Split train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n" + "=" * 60)
    print("STEP 4: Training Model")
    print("=" * 60)
    model = RULModel(model_type='xgboost')
    model.train(X_train, y_train, X_val, y_val)
    model.evaluate(X_val, y_val)
    
    # Save artifacts
    model.save_model('rul_model.pkl')
    prep.save_scaler('scaler.pkl')
    
    print("\n" + "=" * 60)
    print("STEP 5: Loading and Processing Test Data")
    print("=" * 60)
    test_df = loader.load_data(test_file)
    test_df = engineer.engineer_all_features(test_df, simple=simple_features)
    
    print("\n" + "=" * 60)
    print("STEP 6: Making Predictions")
    print("=" * 60)
    pred_engine = PredictionEngine()
    pred_engine.load_artifacts()
    
    # Predict all engines
    results = pred_engine.predict_all_engines(test_df)
    print("\nPrediction Results:")
    print(results.head(10))
    
    # Save results
    pred_engine.save_predictions(results, 'predictions.csv')
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    return pred_engine, results, test_df


# Example usage
if __name__ == "__main__":
    # Run complete pipeline
    pred_engine, results, test_df = run_complete_pipeline(
        train_file='train_FD001.txt',
        test_file='test_FD001.txt',
        simple_features=True
    )
    
    # Example: Get health graph for Engine #5 (as judges might request)
    print("\n" + "=" * 60)
    print("Example: Health Graph for Engine #5")
    print("=" * 60)
    graph_data = pred_engine.generate_health_graph_data(8, test_df)
    print(f"Engine ID: {graph_data['engine_id']}")
    print(f"Current Health: {graph_data['current_health']:.2f}%")
    print(f"Current RUL: {graph_data['current_rul']:.2f} cycles")
    print(f"Status: {graph_data['current_status']}")
    print(f"Recommendation: {graph_data['recommendation']}")