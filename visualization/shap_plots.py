import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Any, List
import catboost
from pathlib import Path

def plot_shap_values(model: catboost.CatBoost,
                    X_test: np.ndarray,
                    feature_names: List[str],
                    class_names: List[str],
                    output_dir: str = 'results/shap_plots') -> None:
    """
    Generate and save SHAP value plots for each emotion class.
    
    Args:
        model: Trained CatBoost model
        X_test: Test features
        feature_names: List of feature names
        class_names: List of emotion class names
        output_dir: Directory to save the plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For each emotion class
    for class_idx, class_name in enumerate(class_names):
        plt.figure(figsize=(12, 8))
        
        # Summary plot for current class
        shap.summary_plot(
            shap_values[class_idx],
            X_test,
            feature_names=feature_names,
            show=False,
            plot_type="bar",
            title=f'SHAP Feature Importance for {class_name}'
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_importance_{class_name}.png')
        plt.close()
        
        # Detailed SHAP values plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[class_idx],
            X_test,
            feature_names=feature_names,
            show=False,
            title=f'SHAP Values Distribution for {class_name}'
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_distribution_{class_name}.png')
        plt.close()

def plot_shap_dependence(model: catboost.CatBoost,
                        X_test: np.ndarray,
                        feature_names: List[str],
                        class_names: List[str],
                        output_dir: str = 'results/shap_plots') -> None:
    """
    Generate and save SHAP dependence plots for the most important features.
    
    Args:
        model: Trained CatBoost model
        X_test: Test features
        feature_names: List of feature names
        class_names: List of emotion class names
        output_dir: Directory to save the plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # For each emotion class
    for class_idx, class_name in enumerate(class_names):
        # Get feature importance based on mean absolute SHAP values
        feature_importance = np.mean(np.abs(shap_values[class_idx]), axis=0)
        top_features = np.argsort(feature_importance)[-5:]  # Top 5 features
        
        for feature_idx in top_features:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                shap_values[class_idx],
                X_test,
                feature_names=feature_names,
                show=False,
                title=f'SHAP Dependence Plot\n{feature_names[feature_idx]} for {class_name}'
            )
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_dependence_{class_name}_{feature_names[feature_idx]}.png')
            plt.close()

# Example usage:
"""
# Assuming you have:
# - modelcb: trained CatBoost model
# - X_test: test features
# - feature_names: list of feature names
# - class_names: list of emotion class names

plot_shap_values(modelcb, X_test, feature_names, class_names)
plot_shap_dependence(modelcb, X_test, feature_names, class_names)
""" 