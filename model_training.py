import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class RULModel:
    """
    Trains and evaluates models for RUL prediction.
    Supports multiple algorithms.
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize model based on type.
        
        Model choices:
        - 'xgboost': Fast, accurate, handles non-linear patterns well
        - 'random_forest': Robust, good for feature importance
        - 'gradient_boosting': Similar to XGBoost but different implementation
        - 'linear': Simple baseline (usually underperforms)
        
        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Create the model instance based on type.
        """
        if self.model_type == 'xgboost':
            # XGBoost: Gradient boosting that's optimized for speed and performance
            return xgb.XGBRegressor(
                n_estimators=200,      # Number of trees
                max_depth=7,            # Maximum depth of each tree
                learning_rate=0.1,      # Step size for weight updates
                subsample=0.8,          # Fraction of samples used per tree
                colsample_bytree=0.8,   # Fraction of features used per tree
                random_state=42,
                n_jobs=-1               # Use all CPU cores
            )
        
        elif self.model_type == 'random_forest':
            # Random Forest: Ensemble of decision trees
            return RandomForestRegressor(
                n_estimators=200,       # Number of trees
                max_depth=15,           # Maximum depth
                min_samples_split=5,    # Minimum samples to split a node
                min_samples_leaf=2,     # Minimum samples in leaf node
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'gradient_boosting':
            # Gradient Boosting: Sequential tree building
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        elif self.model_type == 'linear':
            # Linear Regression: Simple baseline
            return LinearRegression()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on data.
        
        Args:
            X_train: Training features
            y_train: Training targets (RUL)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history (if applicable)
        """
        print(f"Training {self.model_type} model...")
        print(f"Training samples: {len(X_train)}")
        
        if self.model_type == 'xgboost' and X_val is not None:
            # XGBoost: try the older sklearn API first, fallback to callbacks for
            # newer xgboost versions that removed the `early_stopping_rounds` arg.
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )
            except TypeError:
                # If `early_stopping_rounds` isn't accepted, try a plain fit
                # without early stopping (ensures compatibility across xgboost
                # versions). Users can adjust to use `xgb.train` directly
                # for more control if desired.
                self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Training MAE: {train_mae:.2f}")
        
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            print(f"Validation RMSE: {val_rmse:.2f}")
            print(f"Validation MAE: {val_mae:.2f}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted RUL values
        """
        predictions = self.model.predict(X)
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: True RUL values
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1))) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        print("\n=== Model Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get feature importance from tree-based models.
        
        Why: Understanding which sensors/features are most important helps
        explain the model and can guide maintenance strategies.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("This model doesn't support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def save_model(self, filepath='rul_model.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='rul_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


class EnsembleModel:
    """
    Combines multiple models for better predictions.
    
    Why ensemble: Different models capture different patterns.
    Averaging their predictions often gives better results than any single model.
    """
    
    def __init__(self, model_types=['xgboost', 'random_forest']):
        """
        Initialize ensemble with multiple model types.
        
        Args:
            model_types: List of model types to include
        """
        self.models = [RULModel(model_type=mt) for mt in model_types]
        self.weights = None  # Can be set for weighted averaging
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all models in the ensemble.
        """
        print("Training ensemble models...")
        for i, model in enumerate(self.models):
            print(f"\n--- Model {i+1}/{len(self.models)} ---")
            model.train(X_train, y_train, X_val, y_val)
    
    def predict(self, X):
        """
        Make predictions by averaging all model predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Averaged predictions
        """
        predictions = np.array([model.predict(X) for model in self.models])
        
        if self.weights is None:
            # Simple average
            return np.mean(predictions, axis=0)
        else:
            # Weighted average
            return np.average(predictions, axis=0, weights=self.weights)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate ensemble performance.
        """
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        print("\n=== Ensemble Evaluation ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics


# Example usage
if __name__ == "__main__":
    from data_preprocessing import DataLoader
    from feature_engineering import FeatureEngineer
    from data_preparation import DataPreparation
    from sklearn.model_selection import train_test_split
    
    # Load and prepare data
    loader = DataLoader()
    train_df = loader.load_data('train_FD001.txt')
    train_df = loader.add_rul(train_df)
    
    engineer = FeatureEngineer()
    train_df = engineer.engineer_all_features(train_df, simple=True)
    
    prep = DataPreparation()
    X, y = prep.prepare_for_ml(train_df, is_train=True)
    y = prep.clip_rul(y, max_rul=125)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RULModel(model_type='xgboost')
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    model.evaluate(X_val, y_val)
    
    # Save model
    model.save_model()


