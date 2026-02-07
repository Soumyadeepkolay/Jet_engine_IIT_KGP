import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelEvaluator:
    """
    Comprehensive evaluation of RUL prediction models.
    Provides multiple metrics and visualizations to understand model performance.
    """
    
    def __init__(self):
        pass
    
    def calculate_rmse(self, y_true, y_pred):
        """
        Calculate Root Mean Square Error - PRIMARY METRIC for hackathon.
        
        Why RMSE: Penalizes large errors more heavily than MAE, which is important
        for safety-critical applications like predictive maintenance.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            RMSE value
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse
    
    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        Why MAE: Easy to interpret - on average, predictions are off by this many cycles.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            MAE value
        """
        mae = mean_absolute_error(y_true, y_pred)
        return mae
    
    def calculate_mape(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error.
        
        Why MAPE: Shows error as a percentage, useful for understanding
        relative accuracy across different RUL ranges.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            MAPE value (percentage)
        """
        # Add small constant to avoid division by zero
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
        return mape
    
    def calculate_r2_score(self, y_true, y_pred):
        """
        Calculate R² (coefficient of determination).
        
        Why R²: Shows how much variance in RUL is explained by the model.
        1.0 = perfect, 0.0 = as good as predicting the mean.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            R² value
        """
        r2 = r2_score(y_true, y_pred)
        return r2
    
    def calculate_scoring_function(self, y_true, y_pred):
        """
        Calculate NASA's scoring function for RUL prediction.
        
        This is a custom metric used in some competitions that
        penalizes late predictions more than early predictions.
        
        Formula:
        - If predicted < actual: error = exp(-error/13) - 1
        - If predicted >= actual: error = exp(error/10) - 1
        
        Why asymmetric: Better to predict failure too early (safer)
        than too late (dangerous).
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Scoring function value (lower is better)
        """
        d = y_pred - y_true
        score = np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))
        return score
    
    def evaluate_comprehensive(self, y_true, y_pred):
        """
        Calculate all metrics at once.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'RMSE': self.calculate_rmse(y_true, y_pred),
            'MAE': self.calculate_mae(y_true, y_pred),
            'MAPE': self.calculate_mape(y_true, y_pred),
            'R2': self.calculate_r2_score(y_true, y_pred),
            'NASA_Score': self.calculate_scoring_function(y_true, y_pred)
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Pretty print evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<20} {value:.4f}")
        
        print("=" * 50)
    
    def analyze_error_distribution(self, y_true, y_pred):
        """
        Analyze the distribution of prediction errors.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            
        Returns:
            Dictionary with error statistics
        """
        errors = y_pred - y_true
        
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'q25_error': np.percentile(errors, 25),
            'q75_error': np.percentile(errors, 75)
        }
        
        print("\n" + "=" * 50)
        print("ERROR DISTRIBUTION ANALYSIS")
        print("=" * 50)
        print(f"Mean Error (bias): {error_stats['mean_error']:.2f} cycles")
        print(f"Std Dev: {error_stats['std_error']:.2f} cycles")
        print(f"Median Error: {error_stats['median_error']:.2f} cycles")
        print(f"Error Range: [{error_stats['min_error']:.2f}, {error_stats['max_error']:.2f}]")
        print(f"IQR (Q25-Q75): [{error_stats['q25_error']:.2f}, {error_stats['q75_error']:.2f}]")
        print("=" * 50)
        
        # Interpretation
        if abs(error_stats['mean_error']) < 5:
            print("✓ Model is well-calibrated (low bias)")
        elif error_stats['mean_error'] > 5:
            print("⚠ Model tends to over-predict RUL (optimistic)")
        else:
            print("⚠ Model tends to under-predict RUL (pessimistic)")
        
        return error_stats
    
    def analyze_by_rul_range(self, y_true, y_pred, ranges=[(0, 30), (30, 70), (70, 150)]):
        """
        Analyze performance across different RUL ranges.
        
        Why: Models often perform differently when RUL is low vs high.
        Understanding this helps improve the model and set expectations.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            ranges: List of (min, max) tuples for RUL ranges
            
        Returns:
            DataFrame with metrics by range
        """
        results = []
        
        for r_min, r_max in ranges:
            mask = (y_true >= r_min) & (y_true < r_max)
            
            if np.sum(mask) == 0:
                continue
            
            y_true_range = y_true[mask]
            y_pred_range = y_pred[mask]
            
            rmse = self.calculate_rmse(y_true_range, y_pred_range)
            mae = self.calculate_mae(y_true_range, y_pred_range)
            mape = self.calculate_mape(y_true_range, y_pred_range)
            count = len(y_true_range)
            
            results.append({
                'RUL_Range': f"{r_min}-{r_max}",
                'Count': count,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })
        
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 60)
        print("PERFORMANCE BY RUL RANGE")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)
        
        return df
    
    def calculate_prediction_intervals(self, y_true, y_pred, confidence=0.95):
        """
        Calculate confidence intervals for predictions.
        
        Why: Provides uncertainty estimates - important for risk management.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Lower and upper bounds for predictions
        """
        errors = y_pred - y_true
        error_std = np.std(errors)
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)
        
        lower_bound = y_pred - (z * error_std)
        upper_bound = y_pred + (z * error_std)
        
        # Ensure non-negative
        lower_bound = np.maximum(lower_bound, 0)
        
        print(f"\n{confidence*100}% Prediction Intervals:")
        print(f"Average interval width: ±{z * error_std:.2f} cycles")
        
        return lower_bound, upper_bound
    
    def identify_worst_predictions(self, y_true, y_pred, engine_ids=None, top_n=10):
        """
        Identify engines with worst predictions.
        
        Why: Understanding failure cases helps improve the model.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            engine_ids: Array of engine IDs (optional)
            top_n: Number of worst cases to show
            
        Returns:
            DataFrame with worst predictions
        """
        errors = np.abs(y_pred - y_true)
        
        data = {
            'Actual_RUL': y_true,
            'Predicted_RUL': y_pred,
            'Absolute_Error': errors
        }
        
        if engine_ids is not None:
            data['Engine_ID'] = engine_ids
        
        df = pd.DataFrame(data)
        worst = df.nlargest(top_n, 'Absolute_Error')
        
        print("\n" + "=" * 60)
        print(f"TOP {top_n} WORST PREDICTIONS")
        print("=" * 60)
        print(worst.to_string(index=False))
        print("=" * 60)
        
        return worst
    
    def generate_evaluation_report(self, y_true, y_pred, engine_ids=None):
        """
        Generate complete evaluation report.
        
        This is what you should show to judges to demonstrate understanding.
        
        Args:
            y_true: Actual RUL values
            y_pred: Predicted RUL values
            engine_ids: Engine IDs (optional)
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "█" * 70)
        print(" " * 20 + "COMPLETE EVALUATION REPORT")
        print("█" * 70)
        
        # Overall metrics
        metrics = self.evaluate_comprehensive(y_true, y_pred)
        self.print_metrics(metrics)
        
        # Error distribution
        error_stats = self.analyze_error_distribution(y_true, y_pred)
        
        # Performance by RUL range
        range_analysis = self.analyze_by_rul_range(y_true, y_pred)
        
        # Worst predictions
        worst_cases = self.identify_worst_predictions(y_true, y_pred, engine_ids)
        
        # Prediction intervals
        lower, upper = self.calculate_prediction_intervals(y_true, y_pred)
        
        report = {
            'metrics': metrics,
            'error_stats': error_stats,
            'range_analysis': range_analysis,
            'worst_cases': worst_cases
        }
        
        print("\n" + "█" * 70)
        print(" " * 25 + "REPORT COMPLETE")
        print("█" * 70)
        
        return report


# Example usage
if __name__ == "__main__":
    # Simulate predictions for demonstration
    np.random.seed(42)
    
    # Generate synthetic test data
    n_samples = 100
    y_true = np.random.uniform(0, 125, n_samples)
    
    # Simulate predictions with some error
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
    
    engine_ids = np.arange(1, n_samples + 1)
    
    # Run evaluation
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(y_true, y_pred, engine_ids)