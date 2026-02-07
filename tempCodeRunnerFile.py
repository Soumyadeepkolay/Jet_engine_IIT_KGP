import numpy as np
import pandas as pd

class HealthScoreCalculator:
    """
    Converts RUL predictions into intuitive health scores and status levels.
    This makes the predictions actionable for maintenance operators.
    """
    
    def __init__(self, max_rul=125, critical_threshold=30, warning_threshold=70):
        """
        Initialize health calculator with thresholds.
        
        Args:
            max_rul: Maximum RUL value (considered as 100% health)
            critical_threshold: Health % below this is CRITICAL (Red)
            warning_threshold: Health % below this is WARNING (Yellow)
        """
        self.max_rul = max_rul
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
    
    def rul_to_health_percentage(self, rul_value):
        """
        Convert RUL to health percentage.
        
        Formula: Health% = (Current_RUL / Max_RUL) × 100
        
        Logic:
        - RUL = 125 cycles → Health = 100% (brand new)
        - RUL = 62.5 cycles → Health = 50% (half-life)
        - RUL = 0 cycles → Health = 0% (failed)
        
        Args:
            rul_value: Predicted RUL in cycles
            
        Returns:
            Health percentage (0-100)
        """
        health_pct = (rul_value / self.max_rul) * 100
        
        # Clip to valid range [0, 100]
        health_pct = np.clip(health_pct, 0, 100)
        
        return health_pct
    
    def get_health_status(self, health_percentage):
        """
        Determine health status level based on percentage.
        
        Status levels:
        - HEALTHY (Green): 70-100% - No immediate action needed
        - WARNING (Yellow): 30-70% - Schedule maintenance soon
        - CRITICAL (Red): 0-30% - Immediate maintenance required
        
        Args:
            health_percentage: Health percentage value
            
        Returns:
            Status string
        """
        if health_percentage >= self.warning_threshold:
            return "HEALTHY"
        elif health_percentage >= self.critical_threshold:
            return "WARNING"
        else:
            return "CRITICAL"
    
    def get_status_color(self, status):
        """
        Get color code for visualization.
        
        Args:
            status: Health status string
            
        Returns:
            Color name
        """
        color_map = {
            "HEALTHY": "green",
            "WARNING": "yellow",
            "CRITICAL": "red"
        }
        return color_map.get(status, "gray")
    
    def get_maintenance_recommendation(self, health_percentage, rul_value):
        """
        Provide actionable maintenance recommendation.
        
        Args:
            health_percentage: Current health %
            rul_value: Predicted RUL
            
        Returns:
            Recommendation string
        """
        status = self.get_health_status(health_percentage)
        
        if status == "HEALTHY":
            return f"Engine is healthy. Estimated {int(rul_value)} cycles remaining. Continue normal operations."
        
        elif status == "WARNING":
            return f"Maintenance recommended within {int(rul_value)} cycles. Schedule inspection to prevent unexpected failure."
        
        else:  # CRITICAL
            if rul_value < 10:
                return f"URGENT: Only {int(rul_value)} cycles remaining. Immediate maintenance required to avoid failure."
            else:
                return f"Engine health is critical. Schedule maintenance within {int(rul_value)} cycles."
    
    def calculate_degradation_rate(self, rul_history):
        """
        Calculate how fast the engine is degrading.
        
        Why: A rapidly declining health score indicates accelerating degradation,
        which might require more urgent attention than the raw RUL suggests.
        
        Args:
            rul_history: List of RUL values over time (most recent last)
            
        Returns:
            Degradation rate (cycles per time unit)
        """
        if len(rul_history) < 2:
            return 0
        
        # Calculate rate of RUL decrease
        rul_array = np.array(rul_history)
        
        # Fit linear regression to get degradation slope
        time_steps = np.arange(len(rul_array))
        coefficients = np.polyfit(time_steps, rul_array, 1)
        degradation_rate = -coefficients[0]  # Negative because RUL decreases
        
        return degradation_rate
    
    def create_health_report(self, engine_id, rul_prediction, rul_history=None):
        """
        Generate comprehensive health report for an engine.
        
        Args:
            engine_id: Engine identifier
            rul_prediction: Current RUL prediction
            rul_history: Historical RUL values (optional)
            
        Returns:
            Dictionary with complete health information
        """
        # Calculate health metrics
        health_pct = self.rul_to_health_percentage(rul_prediction)
        status = self.get_health_status(health_pct)
        color = self.get_status_color(status)
        recommendation = self.get_maintenance_recommendation(health_pct, rul_prediction)
        
        # Calculate degradation rate if history available
        degradation_rate = None
        if rul_history is not None and len(rul_history) > 1:
            degradation_rate = self.calculate_degradation_rate(rul_history)
        
        report = {
            'engine_id': engine_id,
            'rul_cycles': round(rul_prediction, 2),
            'health_percentage': round(health_pct, 2),
            'status': status,
            'status_color': color,
            'recommendation': recommendation,
            'degradation_rate': round(degradation_rate, 3) if degradation_rate else None
        }
        
        return report
    
    def batch_health_scores(self, engine_ids, rul_predictions):
        """
        Calculate health scores for multiple engines.
        
        Args:
            engine_ids: List of engine IDs
            rul_predictions: List of RUL predictions
            
        Returns:
            DataFrame with health information for all engines
        """
        reports = []
        
        for engine_id, rul_pred in zip(engine_ids, rul_predictions):
            report = self.create_health_report(engine_id, rul_pred)
            reports.append(report)
        
        df = pd.DataFrame(reports)
        
        # Sort by health percentage (worst first)
        df = df.sort_values('health_percentage')
        
        return df
    
    def get_priority_engines(self, health_df, top_n=10):
        """
        Identify engines that need immediate attention.
        
        Args:
            health_df: DataFrame with health scores
            top_n: Number of top priority engines to return
            
        Returns:
            DataFrame of engines requiring priority maintenance
        """
        # Filter critical and warning engines
        priority_df = health_df[health_df['status'].isin(['CRITICAL', 'WARNING'])]
        
        # Sort by health percentage (worst first)
        priority_df = priority_df.sort_values('health_percentage').head(top_n)
        
        print(f"\n=== Top {len(priority_df)} Priority Engines ===")
        print(priority_df[['engine_id', 'health_percentage', 'rul_cycles', 'status']].to_string(index=False))
        
        return priority_df


class HealthVisualizer:
    """
    Creates visualization data for health scores and trends.
    (This prepares data; actual plotting happens in dashboard)
    """
    
    def __init__(self):
        self.calculator = HealthScoreCalculator()
    
    def prepare_health_timeline(self, engine_data, rul_predictions):
        """
        Prepare data for health timeline visualization.
        
        Args:
            engine_data: DataFrame with time_cycles column
            rul_predictions: Array of RUL predictions for each cycle
            
        Returns:
            DataFrame ready for plotting
        """
        timeline_data = pd.DataFrame({
            'cycle': engine_data['time_cycles'].values,
            'rul': rul_predictions,
            'health_pct': [self.calculator.rul_to_health_percentage(rul) for rul in rul_predictions],
            'status': [self.calculator.get_health_status(
                self.calculator.rul_to_health_percentage(rul)
            ) for rul in rul_predictions]
        })
        
        return timeline_data
    
    def prepare_gauge_data(self, health_percentage):
        """
        Prepare data for gauge chart visualization.
        
        Args:
            health_percentage: Current health percentage
            
        Returns:
            Dictionary with gauge configuration
        """
        status = self.calculator.get_health_status(health_percentage)
        color = self.calculator.get_status_color(status)
        
        gauge_data = {
            'value': health_percentage,
            'status': status,
            'color': color,
            'ranges': [
                {'range': [0, 30], 'color': 'red', 'label': 'Critical'},
                {'range': [30, 70], 'color': 'yellow', 'label': 'Warning'},
                {'range': [70, 100], 'color': 'green', 'label': 'Healthy'}
            ]
        }
        
        return gauge_data


# Example usage
if __name__ == "__main__":
    # Example: Calculate health for different RUL values
    calculator = HealthScoreCalculator(max_rul=125)
    
    # Test cases
    test_ruls = [125, 100, 75, 50, 30, 15, 5, 0]
    
    print("=== Health Score Examples ===\n")
    for rul in test_ruls:
        health_pct = calculator.rul_to_health_percentage(rul)
        status = calculator.get_health_status(health_pct)
        recommendation = calculator.get_maintenance_recommendation(health_pct, rul)
        
        print(f"RUL: {rul} cycles")
        print(f"Health: {health_pct:.1f}%")
        print(f"Status: {status}")
        print(f"Recommendation: {recommendation}")
        print("-" * 60)
    
    # Example: Full health report
    print("\n=== Full Health Report ===")
    report = calculator.create_health_report(
        engine_id=8,
        rul_prediction=45.5,
        rul_history=[80, 75, 68, 60, 50, 45.5]
    )
    
    for key, value in report.items():
        print(f"{key}: {value}")