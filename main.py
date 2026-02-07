"""
Main execution script for the Predictive Maintenance Hackathon.

This script orchestrates the entire pipeline and provides a simple interface
to train models and generate predictions for the dashboard.

Usage:
    python main.py --mode train --train_file train_FD001.txt
    python main.py --mode predict --test_file test_FD001.txt --engine_id 5
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import all custom modules
from data_preprocessing import DataLoader
from feature_engineering import FeatureEngineer
from data_preparation import DataPreparation
from model_training import RULModel, EnsembleModel
from health_score_calculator import HealthScoreCalculator
from prediction_engine import PredictionEngine
from evaluation_metrics import ModelEvaluator

class HackathonPipeline:
    """
    Main pipeline controller for the hackathon project.
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.data_prep = DataPreparation()
        self.health_calculator = HealthScoreCalculator(max_rul=125)
        self.evaluator = ModelEvaluator()
        
        self.model = None
        self.train_df = None
        self.test_df = None
    
    def train_model(self, train_file, model_type='xgboost', simple_features=True):
        """
        Complete training pipeline.
        
        Args:
            train_file: Path to training data file
            model_type: Type of model to train
            simple_features: Use simple feature engineering
            
        Returns:
            Trained model
        """
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        print("\n📂 Step 1/5: Loading training data...")
        self.train_df = self.data_loader.load_data(train_file)
        self.train_df = self.data_loader.add_rul(self.train_df)
        print(f"✓ Loaded {len(self.train_df)} training samples from {self.train_df['unit_id'].nunique()} engines")
        
        # Step 2: Engineer features
        print("\n🔧 Step 2/5: Engineering features...")
        self.train_df = self.feature_engineer.engineer_all_features(
            self.train_df, 
            simple=simple_features
        )
        print(f"✓ Created {len(self.train_df.columns)} total features")
        
        # Step 3: Prepare data
        print("\n📊 Step 3/5: Preparing data for training...")
        X, y = self.data_prep.prepare_for_ml(self.train_df, is_train=True)
        y = self.data_prep.clip_rul(y, max_rul=125)
        
        # Split train/validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✓ Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Step 4: Train model
        print(f"\n🤖 Step 4/5: Training {model_type} model...")
        self.model = RULModel(model_type=model_type)
        self.model.train(X_train, y_train, X_val, y_val)
        
        # Step 5: Evaluate
        print("\n📈 Step 5/5: Evaluating model...")
        val_pred = self.model.predict(X_val)
        metrics = self.evaluator.evaluate_comprehensive(y_val, val_pred)
        self.evaluator.print_metrics(metrics)
        
        # Feature importance
        if hasattr(self.model.model, 'feature_importances_'):
            print("\n🎯 Feature Importance Analysis:")
            self.model.get_feature_importance(self.data_prep.feature_columns, top_n=10)
        
        # Save artifacts
        print("\n💾 Saving model and scaler...")
        self.model.save_model('rul_model.pkl')
        self.data_prep.save_scaler('scaler.pkl')
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"📊 Validation RMSE: {metrics['RMSE']:.2f} cycles")
        print(f"📁 Model saved to: rul_model.pkl")
        print(f"📁 Scaler saved to: scaler.pkl")
        print("="*70)
        
        return self.model
    
    def predict_test_set(self, test_file, simple_features=True):
        """
        Make predictions on test set.
        
        Args:
            test_file: Path to test data file
            simple_features: Use simple feature engineering
            
        Returns:
            DataFrame with predictions
        """
        print("\n" + "="*70)
        print("🔮 STARTING PREDICTION PIPELINE")
        print("="*70)
        
        # Load test data
        print("\n📂 Step 1/3: Loading test data...")
        self.test_df = self.data_loader.load_data(test_file, has_rul=False)
        print(f"✓ Loaded data for {self.test_df['unit_id'].nunique()} engines")
        
        # Engineer features
        print("\n🔧 Step 2/3: Engineering features...")
        self.test_df = self.feature_engineer.engineer_all_features(
            self.test_df,
            simple=simple_features
        )
        
        # Make predictions
        print("\n🎯 Step 3/3: Making predictions...")
        pred_engine = PredictionEngine()
        pred_engine.load_artifacts()
        
        results_df = pred_engine.predict_all_engines(self.test_df)
        
        print("\n" + "="*70)
        print("✅ PREDICTIONS COMPLETE!")
        print("="*70)
        print(f"📊 Predicted RUL for {len(results_df)} engines")
        print("\n📋 Summary Statistics:")
        print(f"  Average Health: {results_df['health_percentage'].mean():.1f}%")
        print(f"  Critical Engines: {len(results_df[results_df['status'] == 'CRITICAL'])}")
        print(f"  Warning Engines: {len(results_df[results_df['status'] == 'WARNING'])}")
        print(f"  Healthy Engines: {len(results_df[results_df['status'] == 'HEALTHY'])}")
        print("="*70)
        
        # Save results
        results_df.to_csv('predictions.csv', index=False)
        print("\n💾 Results saved to: predictions.csv")
        
        return results_df
    
    def predict_single_engine(self, test_file, engine_id, simple_features=True):
        """
        Make prediction for a specific engine and generate health graph data.
        This is what you'll demo to the judges.
        
        Args:
            test_file: Path to test data file
            engine_id: Engine ID to predict
            simple_features: Use simple feature engineering
            
        Returns:
            Dictionary with complete health information
        """
        print("\n" + "="*70)
        print(f"🎯 PREDICTING ENGINE #{engine_id}")
        print("="*70)
        
        # Load and prepare test data if not already done
        if self.test_df is None:
            print("\n📂 Loading test data...")
            self.test_df = self.data_loader.load_data(test_file, has_rul=False)
            self.test_df = self.feature_engineer.engineer_all_features(
                self.test_df,
                simple=simple_features
            )
        
        # Make prediction
        pred_engine = PredictionEngine()
        pred_engine.load_artifacts()
        
        # Get complete health graph data
        graph_data = pred_engine.generate_health_graph_data(engine_id, self.test_df)
        
        # Display results
        print("\n" + "="*70)
        print("📊 PREDICTION RESULTS")
        print("="*70)
        print(f"Engine ID: {graph_data['engine_id']}")
        print(f"Current Health: {graph_data['current_health']:.2f}%")
        print(f"Remaining Useful Life: {graph_data['current_rul']:.2f} cycles")
        print(f"Status: {graph_data['current_status']}")
        print(f"\n💡 Recommendation:")
        print(f"  {graph_data['recommendation']}")
        print("="*70)
        
        return graph_data
    
    def quick_demo(self, train_file='train_FD001.txt', test_file='test_FD001.txt'):
        """
        Quick demo for hackathon - trains model and shows predictions.
        
        Args:
            train_file: Training data file
            test_file: Test data file
        """
        print("\n" + "█"*70)
        print(" "*15 + "PREDICTIVE MAINTENANCE DEMO")
        print("█"*70)
        
        # Train
        self.train_model(train_file, model_type='xgboost', simple_features=True)
        
        # Predict all
        results = self.predict_test_set(test_file, simple_features=True)
        
        # Show example engine
        print("\n" + "█"*70)
        print("📊 EXAMPLE: Health Graph for Engine #5")
        print("█"*70)
        graph_data = self.predict_single_engine(test_file, engine_id=5)
        
        # Show priority engines
        print("\n" + "█"*70)
        print("⚠️  PRIORITY MAINTENANCE SCHEDULE")
        print("█"*70)
        priority = results[results['status'] != 'HEALTHY'].sort_values('health_percentage').head(5)
        print(priority[['engine_id', 'health_percentage', 'rul_cycles', 'status']].to_string(index=False))
        
        print("\n" + "█"*70)
        print("✅ DEMO COMPLETE - Ready for Dashboard!")
        print("█"*70)


def main():
    """
    Command-line interface for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Predictive Maintenance System for Jet Engines'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'demo'],
        default='demo',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        default='train_FD001.txt',
        help='Path to training data file'
    )
    
    parser.add_argument(
        '--test_file',
        type=str,
        default='test_FD001.txt',
        help='Path to test data file'
    )
    
    parser.add_argument(
        '--engine_id',
        type=int,
        help='Specific engine ID to predict (for predict mode)'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'gradient_boosting'],
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--simple_features',
        action='store_true',
        default=True,
        help='Use simple feature engineering (faster)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = HackathonPipeline()
    
    # Execute based on mode
    if args.mode == 'train':
        pipeline.train_model(
            args.train_file,
            model_type=args.model_type,
            simple_features=args.simple_features
        )
    
    elif args.mode == 'predict':
        if args.engine_id:
            pipeline.predict_single_engine(
                args.test_file,
                args.engine_id,
                simple_features=args.simple_features
            )
        else:
            pipeline.predict_test_set(
                args.test_file,
                simple_features=args.simple_features
            )
    
    elif args.mode == 'demo':
        pipeline.quick_demo(args.train_file, args.test_file)


if __name__ == "__main__":
    # If run without arguments, show demo
    if len(sys.argv) == 1:
        print("\n💡 Running in DEMO mode...")
        print("For other modes, use: python main.py --mode [train|predict|demo]")
        
        pipeline = HackathonPipeline()
        pipeline.quick_demo()
    else:
        main()