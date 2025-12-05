"""
Machine Learning Pipeline System for Production AI Applications

This implementation provides an end-to-end ML pipeline system including
data preprocessing, model training, evaluation, deployment, and monitoring.

Author: Satbir Singh
Paper: IEEE Conference Paper
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Optional, Tuple
import pickle
import json
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class ModelVersion:
    """Container for model version information"""
    version: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    features: List[str]
    hyperparameters: Dict


class DataPreprocessor:
    """
    Handles data preprocessing including cleaning, scaling, and feature engineering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform data"""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        return self.scaler.transform(X)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for training"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Remove outliers (simplified)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df


class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and evaluation
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.training_history = []
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, **hyperparameters) -> Dict:
        """
        Train model with data preprocessing and evaluation
        """
        # Clean data
        X_clean = self.preprocessor.clean_data(X.copy())
        y_clean = y.loc[X_clean.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=test_size, random_state=42
        )
        
        # Preprocess
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=hyperparameters.get('n_estimators', 100),
                max_depth=hyperparameters.get('max_depth', None),
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, test_pred, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, test_pred, average='weighted', zero_division=0),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Store training history
        version = ModelVersion(
            version=f"v{len(self.training_history) + 1}.0",
            timestamp=datetime.now(),
            accuracy=metrics['test_accuracy'],
            precision=metrics['test_precision'],
            recall=metrics['test_recall'],
            f1_score=metrics['test_f1'],
            training_samples=metrics['training_samples'],
            features=self.preprocessor.feature_names,
            hyperparameters=hyperparameters
        )
        self.training_history.append(version)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importance_dict = {
            feature: importance 
            for feature, importance in zip(
                self.preprocessor.feature_names,
                self.model.feature_importances_
            )
        }
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath: str):
        """Save trained model and preprocessor"""
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'training_history': [asdict(v) for v in self.training_history]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load trained model and preprocessor"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.model_type = model_data['model_type']
        self.training_history = [
            ModelVersion(**v) for v in model_data['training_history']
        ]


class MLPipeline:
    """
    End-to-end ML pipeline orchestrator
    """
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.trainer = ModelTrainer()
        self.deployment_status = 'not_deployed'
        self.deployment_timestamp = None
    
    def run_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                    hyperparameters: Optional[Dict] = None) -> Dict:
        """
        Execute complete ML pipeline: preprocessing -> training -> evaluation
        """
        print(f"Running ML Pipeline: {self.pipeline_name}")
        print("-" * 60)
        
        # Training phase
        print("Phase 1: Training Model...")
        metrics = self.trainer.train(
            X, y, 
            **(hyperparameters or {})
        )
        
        print(f"  ✓ Training completed")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
        print(f"  Test F1-Score: {metrics['test_f1']:.2%}")
        print()
        
        # Feature importance
        print("Phase 2: Analyzing Feature Importance...")
        importance = self.trainer.get_feature_importance()
        top_features = list(importance.items())[:5]
        print("  Top 5 Features:")
        for feature, score in top_features:
            print(f"    - {feature}: {score:.4f}")
        print()
        
        return {
            'pipeline_name': self.pipeline_name,
            'metrics': metrics,
            'feature_importance': importance,
            'model_version': self.trainer.training_history[-1].version
        }
    
    def deploy(self, model_path: Optional[str] = None):
        """Deploy model to production"""
        if self.trainer.model is None:
            raise ValueError("Model must be trained before deployment")
        
        if model_path:
            self.trainer.save_model(model_path)
        
        self.deployment_status = 'deployed'
        self.deployment_timestamp = datetime.now()
        print(f"✓ Model deployed: {self.pipeline_name}")
        print(f"  Deployment time: {self.deployment_timestamp}")
    
    def predict_batch(self, X: pd.DataFrame) -> Dict:
        """Make batch predictions"""
        if self.deployment_status != 'deployed':
            raise ValueError("Model must be deployed before prediction")
        
        predictions = self.trainer.predict(X)
        probabilities = self.trainer.predict_proba(X)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'sample_count': len(X)
        }
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            'pipeline_name': self.pipeline_name,
            'deployment_status': self.deployment_status,
            'deployment_timestamp': self.deployment_timestamp.isoformat() if self.deployment_timestamp else None,
            'model_versions': len(self.trainer.training_history),
            'latest_version': self.trainer.training_history[-1].version if self.trainer.training_history else None
        }


def generate_sample_data(n_samples: int = 1000, n_features: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i+1}' for i in range(n_features)]
    )
    
    # Generate target (binary classification)
    y = pd.Series(
        (X.sum(axis=1) > 0).astype(int),
        name='target'
    )
    
    return X, y


def main():
    """Demonstration of ML Pipeline System"""
    print("=" * 60)
    print("Machine Learning Pipeline System")
    print("IEEE Conference Paper Implementation")
    print("=" * 60)
    print()
    
    # Generate sample data
    print("Generating sample dataset...")
    X, y = generate_sample_data(n_samples=1000, n_features=10)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # Initialize pipeline
    pipeline = MLPipeline("production_classifier")
    
    # Run pipeline
    results = pipeline.run_pipeline(X, y, hyperparameters={'n_estimators': 100})
    
    # Deploy model
    print("Phase 3: Deploying Model...")
    pipeline.deploy()
    print()
    
    # Make predictions
    print("Phase 4: Making Predictions...")
    test_samples = X.sample(10, random_state=42)
    predictions = pipeline.predict_batch(test_samples)
    print(f"  ✓ Generated predictions for {predictions['sample_count']} samples")
    print()
    
    # Pipeline status
    print("=" * 60)
    print("Pipeline Status")
    print("=" * 60)
    status = pipeline.get_pipeline_status()
    for key, value in status.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()


if __name__ == "__main__":
    main()

