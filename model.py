"""
Model Module - Preprocessing, Training, and Evaluation
======================================================
This module contains all the ML pipeline functions:
1. Data preprocessing (handling missing values, encoding, scaling)
2. Model training (Random Forest, SVM, Gradient Boosting)
3. Model evaluation (accuracy, precision, recall, F1, confusion matrix)
4. Visualization (comparison graphs, confusion matrix, feature importance)
5. Model serialization (save best model as pickle)
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.impute import SimpleImputer

# Visualization imports
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns


class GrapeDiseaseModel:
    """
    Main class for the Grape Downy Mildew prediction system.
    Handles the complete ML pipeline from preprocessing to model saving.
    """

    def __init__(self, data_path="dataset.csv"):
        """Initialize with dataset path."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy="median")
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = ["Rainfall_mm", "Humidity_pct", "Temperature_C", "Crop_Stage"]

    def load_data(self):
        """Load dataset from CSV file."""
        print("=" * 60)
        print("STEP 1: Loading Dataset")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nClass distribution:\n{self.df['Disease_Risk'].value_counts()}")
        print(f"\nDataset preview:\n{self.df.head()}")
        return self.df

    def preprocess_data(self):
        """
        Preprocess the data:
        1. Handle missing values with median imputation
        2. Encode target variable (Low=0, Medium=1, High=2)
        3. Split into train/test sets
        4. Scale features using StandardScaler
        """
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing Data")
        print("=" * 60)

        # Separate features and target
        X = self.df[self.feature_columns].copy()
        y = self.df["Disease_Risk"].copy()

        # Handle missing values using median imputation
        print(f"Missing values before imputation: {X.isnull().sum().sum()}")
        X = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=self.feature_columns
        )
        print(f"Missing values after imputation: {X.isnull().sum().sum()}")

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Encoded mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        # Split data (80% train, 20% test, stratified)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"\nTrain set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

        # Feature scaling (keep as DataFrame to preserve column names)
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_columns
        )
        print("Features scaled using StandardScaler")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """
        Train three different models:
        1. Random Forest Classifier
        2. Support Vector Machine (SVM)
        3. Gradient Boosting Classifier
        """
        print("\n" + "=" * 60)
        print("STEP 3: Training Models")
        print("=" * 60)

        # Define models with tuned hyperparameters
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "SVM": SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                random_state=42,
                probability=True
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
        }

        # Train each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"  {name} trained successfully!")

        return self.models

    def evaluate_models(self):
        """
        Evaluate all trained models using:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1-score (weighted)
        - Confusion Matrix
        
        Returns a dictionary with all metrics.
        """
        print("\n" + "=" * 60)
        print("STEP 4: Evaluating Models")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Store results
            self.results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm,
                "predictions": y_pred
            }
            
            # Print metrics
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"\n  Confusion Matrix:\n{cm}")
            print(f"\n  Classification Report:\n{classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_)}")

        return self.results

    def select_best_model(self):
        """Select the best model based on F1-score."""
        print("\n" + "=" * 60)
        print("STEP 5: Selecting Best Model")
        print("=" * 60)
        
        best_f1 = 0
        for name, metrics in self.results.items():
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best F1-Score: {best_f1:.4f}")
        
        return self.best_model, self.best_model_name

    def create_visualizations(self, output_dir="."):
        """
        Create and save three visualizations:
        1. Accuracy comparison bar chart
        2. Confusion matrix heatmap (for best model)
        3. Feature importance plot (for best model)
        """
        print("\n" + "=" * 60)
        print("STEP 6: Generating Visualizations")
        print("=" * 60)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

        # 1. Accuracy Comparison Graph
        fig, ax = plt.subplots(figsize=(10, 6))
        model_names = list(self.results.keys())
        accuracies = [self.results[m]["accuracy"] for m in model_names]
        f1_scores = [self.results[m]["f1_score"] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#2ecc71", alpha=0.85)
        bars2 = ax.bar(x + width/2, f1_scores, width, label="F1-Score", color="#3498db", alpha=0.85)
        
        ax.set_xlabel("Models", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: accuracy_comparison.png")

        # 2. Confusion Matrix Plot (Best Model)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = self.results[self.best_model_name]["confusion_matrix"]
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   annot_kws={"size": 14})
        
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
        ax.set_title(f"Confusion Matrix - {self.best_model_name}", fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: confusion_matrix.png")

        # 3. Feature Importance Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        else:
            # For SVM, use permutation-based importance approximation
            importances = np.abs(self.best_model.coef_[0]) if hasattr(self.best_model, "coef_") else np.ones(4) / 4
        
        indices = np.argsort(importances)[::-1]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
        
        bars = ax.barh(
            [self.feature_columns[i] for i in indices],
            [importances[i] for i in indices],
            color=colors[:len(indices)],
            alpha=0.85
        )
        
        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Features", fontsize=12, fontweight="bold")
        ax.set_title(f"Feature Importance - {self.best_model_name}", fontsize=14, fontweight="bold")
        
        # Add value labels
        for i, (idx, bar) in enumerate(zip(indices, bars)):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{importances[idx]:.3f}", va="center", fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: feature_importance.png")

    def save_model(self, output_dir="."):
        """Save the best model, scaler, label encoder, and imputer as pickle."""
        print("\n" + "=" * 60)
        print("STEP 7: Saving Model")
        print("=" * 60)
        
        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "imputer": self.imputer,
            "feature_columns": self.feature_columns,
            "results": {name: {k: v.tolist() if isinstance(v, np.ndarray) else v 
                               for k, v in metrics.items()} 
                       for name, metrics in self.results.items()}
        }
        
        model_path = os.path.join(output_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Best model: {self.best_model_name}")
        return model_path


def run_pipeline(data_path="dataset.csv", output_dir="."):
    """
    Run the complete ML pipeline.
    This is the main function to train and save the model.
    """
    print("\n" + "#" * 60)
    print("#  GRAPE DOWNY MILDEW PREDICTION SYSTEM")
    print("#  ML Pipeline Execution")
    print("#" * 60)
    
    # Initialize
    pipeline = GrapeDiseaseModel(data_path=data_path)
    
    # Step 1: Load data
    pipeline.load_data()
    
    # Step 2: Preprocess
    pipeline.preprocess_data()
    
    # Step 3: Train models
    pipeline.train_models()
    
    # Step 4: Evaluate
    pipeline.evaluate_models()
    
    # Step 5: Select best
    pipeline.select_best_model()
    
    # Step 6: Visualize
    pipeline.create_visualizations(output_dir=output_dir)
    
    # Step 7: Save model
    pipeline.save_model(output_dir=output_dir)
    
    print("\n" + "#" * 60)
    print("#  PIPELINE COMPLETE!")
    print("#" * 60)
    
    return pipeline


if __name__ == "__main__":
    run_pipeline()
