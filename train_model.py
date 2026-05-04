"""
Training Script - Entry Point for Model Training
=================================================
This script runs the complete ML pipeline:
1. Loads the dataset
2. Preprocesses the data
3. Trains multiple models
4. Evaluates and compares them
5. Saves the best model and visualizations

Usage:
    python train_model.py
"""

import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import run_pipeline


def main():
    """Main function to run the training pipeline."""
    
    # Set paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset.csv")
    output_dir = base_dir
    
    # Verify dataset exists
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please run generate_dataset.py first to create the dataset.")
        sys.exit(1)
    
    print(f"Dataset path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Run the complete pipeline
    pipeline = run_pipeline(data_path=data_path, output_dir=output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"\nBest Model: {pipeline.best_model_name}")
    print(f"Best F1-Score: {pipeline.results[pipeline.best_model_name]['f1_score']:.4f}")
    
    print("\nAll Model Results:")
    for name, metrics in pipeline.results.items():
        print(f"  {name}:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nGenerated Files:")
    print("  - model.pkl (trained model)")
    print("  - accuracy_comparison.png")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    
    print("\nYou can now run the Flask app with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
