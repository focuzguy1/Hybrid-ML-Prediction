#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dependencies:
-------------
- Python >= 3.7
- pandas, numpy, scikit-learn, scipy, matplotlib, joblib

Usage:
------
python hybrid_smorff_predictor.py
"""

# ============================================
# IMPORTS
# ============================================
import os
import pickle
import math
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style for publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300


# ============================================
# CONFIGURATION
# ============================================
class Config:
    """Configuration parameters for the model"""
    # File paths
    DATA_PATH = 'Mtb_positive_negative_data.csv'
    MODEL_SAVE_PATH = 'hybrid_smorff_model.pkl'
    RESULTS_DIR = 'results'
    
    # Model hyperparameters (n_estimators = 500)
    RF_N_ESTIMATORS = 500          # Random Forest trees
    RF_MAX_DEPTH = 10
    GB_N_ESTIMATORS = 500          # Gradient Boosting iterations
    GB_MAX_DEPTH = 10
    GB_LEARNING_RATE = 0.1
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    K_MER_SIZE = 6
    
    # Cross-validation
    CV_FOLDS = 10
    
    # Confidence interval
    CONFIDENCE_LEVEL = 0.95


# ============================================
# CONFIDENCE INTERVAL FUNCTIONS
# ============================================

def auc_ci(auc_score, n_positive, n_negative, confidence=Config.CONFIDENCE_LEVEL):
    """
    Calculate confidence interval for AUC using Hanley-McNeil method.
    
    Parameters
    ----------
    auc_score : float
        Area under the ROC curve
    n_positive : int
        Number of positive samples
    n_negative : int
        Number of negative samples
    confidence : float
        Confidence level (default: 0.95)
    
    Returns
    -------
    tuple : (ci_lower, ci_upper, standard_error)
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    q1 = auc_score / (2 - auc_score)
    q2 = (2 * auc_score**2) / (1 + auc_score)
    
    se = math.sqrt((auc_score * (1 - auc_score) + 
                    (n_positive - 1) * (q1 - auc_score**2) + 
                    (n_negative - 1) * (q2 - auc_score**2)) / 
                   (n_positive * n_negative))
    
    ci_lower = max(0, auc_score - z * se)
    ci_upper = min(1, auc_score + z * se)
    
    return ci_lower, ci_upper, se


def accuracy_ci(accuracy, n, confidence=Config.CONFIDENCE_LEVEL):
    """
    Calculate confidence interval for accuracy using Wilson score interval.
    
    Parameters
    ----------
    accuracy : float
        Accuracy score
    n : int
        Number of samples
    confidence : float
        Confidence level (default: 0.95)
    
    Returns
    -------
    tuple : (ci_lower, ci_upper)
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n_success = accuracy * n
    centre = (n_success + z**2 / 2) / n
    half_width = z * math.sqrt((n_success * (n - n_success)) / n + z**2 / 4) / n
    
    ci_lower = max(0, centre - half_width)
    ci_upper = min(1, centre + half_width)
    
    return ci_lower, ci_upper


def mcc_ci(mcc, n, confidence=Config.CONFIDENCE_LEVEL):
    """
    Calculate approximate confidence interval for MCC using normal approximation.
    
    Parameters
    ----------
    mcc : float
        Matthews correlation coefficient
    n : int
        Number of samples
    confidence : float
        Confidence level (default: 0.95)
    
    Returns
    -------
    tuple : (ci_lower, ci_upper)
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = math.sqrt((1 - mcc**2) / (n - 2))
    
    ci_lower = max(-1, mcc - z * se)
    ci_upper = min(1, mcc + z * se)
    
    return ci_lower, ci_upper


# ============================================
# FEATURE EXTRACTION
# ============================================

def get_kmers(sequence, size=Config.K_MER_SIZE):
    """
    Convert DNA sequence into k-mer words.
    
    Parameters
    ----------
    sequence : str
        DNA sequence
    size : int
        k-mer size (default: 6)
    
    Returns
    -------
    list : List of k-mer strings
    """
    sequence = str(sequence).upper().strip()
    if len(sequence) < size:
        return [sequence]
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


def extract_kmer_features(sequences, vectorizer=None, fit=False):
    """
    Extract k-mer features from sequences.
    
    Parameters
    ----------
    sequences : list
        List of DNA sequences
    vectorizer : CountVectorizer, optional
        Pre-fitted vectorizer
    fit : bool
        Whether to fit a new vectorizer
    
    Returns
    -------
    tuple : (features, vectorizer)
    """
    # Convert sequences to space-separated k-mers
    kmer_strings = [' '.join(get_kmers(seq)) for seq in sequences]
    
    if fit or vectorizer is None:
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        features = vectorizer.fit_transform(kmer_strings)
    else:
        features = vectorizer.transform(kmer_strings)
    
    return features, vectorizer


# ============================================
# MODEL BUILDING
# ============================================

def build_hybrid_model(random_state=Config.RANDOM_STATE):
    """
    Build the hybrid ensemble model with n_estimators = 500.
    
    Parameters
    ----------
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    VotingClassifier : Hybrid model
    """
    rf = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=Config.GB_N_ESTIMATORS,
        learning_rate=Config.GB_LEARNING_RATE,
        max_depth=Config.GB_MAX_DEPTH,
        random_state=random_state
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    return ensemble


# ============================================
# MODEL SAVING AND LOADING (OPTIONAL)
# ============================================

def save_model_package(model, vectorizer, filepath=Config.MODEL_SAVE_PATH):
    """
    Save model and vectorizer together in a single file (optional).
    
    Parameters
    ----------
    model : VotingClassifier
        Trained hybrid model
    vectorizer : CountVectorizer
        Fitted vectorizer
    filepath : str
        Path to save the model package
    """
    package = {
        'model': model,
        'vectorizer': vectorizer,
        'model_type': 'VotingClassifier (Random Forest + Gradient Boosting)',
        'n_estimators_rf': Config.RF_N_ESTIMATORS,
        'n_estimators_gb': Config.GB_N_ESTIMATORS,
        'kmer_size': Config.K_MER_SIZE,
        'features': '6-mer frequency',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '2.0'
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(package, f)
    
    print(f" Model package saved to: {filepath}")
    return filepath


def load_model_package(filepath=Config.MODEL_SAVE_PATH):
    """
    Load model and vectorizer from a single file (optional).
    
    Parameters
    ----------
    filepath : str
        Path to the saved model package
    
    Returns
    -------
    tuple : (model, vectorizer)
    """
    with open(filepath, 'rb') as f:
        package = pickle.load(f)
    
    print(f"✓ Model package loaded from: {filepath}")
    print(f"  Model type: {package.get('model_type', 'Unknown')}")
    print(f"  Training date: {package.get('training_date', 'Unknown')}")
    
    return package['model'], package['vectorizer']


# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_model(y_true, y_pred, y_proba):
    """
    Calculate all evaluation metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities
    
    Returns
    -------
    dict : Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Specificity from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc_score = auc(fpr, tpr)
    
    # Sample sizes
    n_pos = sum(y_true == 1)
    n_neg = sum(y_true == 0)
    n_total = len(y_true)
    
    # Confidence intervals
    auc_lower, auc_upper, auc_se = auc_ci(auc_score, n_pos, n_neg)
    acc_lower, acc_upper = accuracy_ci(accuracy, n_total)
    mcc_lower, mcc_upper = mcc_ci(mcc, n_total)
    
    metrics = {
        'accuracy': accuracy,
        'accuracy_ci': (acc_lower, acc_upper),
        'precision': precision,
        'sensitivity': recall,
        'specificity': specificity,
        'f1_score': f1,
        'mcc': mcc,
        'mcc_ci': (mcc_lower, mcc_upper),
        'auc': auc_score,
        'auc_ci': (auc_lower, auc_upper),
        'auc_se': auc_se,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return metrics


def print_results(metrics, n_samples, n_pos, n_neg):
    """
    Print evaluation results in a formatted table.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics from evaluate_model()
    n_samples : int
        Total number of samples
    n_pos : int
        Number of positive samples
    n_neg : int
        Number of negative samples
    """
    print("\n" + "="*70)
    print("HYBRID MODEL PERFORMANCE WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    print(f"\nSample size: {n_samples} sequences ({n_pos} positive, {n_neg} negative)")
    print(f"\n{'Metric':<18} {'Value':<12} {'95% CI':<25} {'SE':<12}")
    print("-"*70)
    
    acc_lower, acc_upper = metrics['accuracy_ci']
    print(f"{'Accuracy':<18} {metrics['accuracy']:<12.4f} [{acc_lower:.4f} - {acc_upper:.4f}] {'N/A':<12}")
    
    auc_lower, auc_upper = metrics['auc_ci']
    print(f"{'AUC':<18} {metrics['auc']:<12.4f} [{auc_lower:.4f} - {auc_upper:.4f}] {metrics['auc_se']:<12.4f}")
    
    mcc_lower, mcc_upper = metrics['mcc_ci']
    print(f"{'MCC':<18} {metrics['mcc']:<12.4f} [{mcc_lower:.4f} - {mcc_upper:.4f}] {'N/A':<12}")
    
    print(f"{'Precision':<18} {metrics['precision']:<12.4f} {'N/A':<25} {'N/A':<12}")
    print(f"{'Sensitivity':<18} {metrics['sensitivity']:<12.4f} {'N/A':<25} {'N/A':<12}")
    print(f"{'Specificity':<18} {metrics['specificity']:<12.4f} {'N/A':<25} {'N/A':<12}")
    print(f"{'F1 Score':<18} {metrics['f1_score']:<12.4f} {'N/A':<25} {'N/A':<12}")
    
    cm = metrics['confusion_matrix']
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print(f"  True Negatives (TN): {cm[0,0]}")
    print(f"  False Positives (FP): {cm[0,1]}")
    print(f"  False Negatives (FN): {cm[1,0]}")
    print(f"  True Positives (TP): {cm[1,1]}")


# ============================================
# VISUALIZATION
# ============================================

def plot_roc_curve(metrics, save_path=None):
    """
    Plot ROC curve with confidence interval.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metrics from evaluate_model()
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(8, 8))
    
    auc_lower, auc_upper = metrics['auc_ci']
    
    plt.plot(metrics['fpr'], metrics['tpr'], color='#1f77b4', lw=2.5,
             label=f'Hybrid Model (AUC = {metrics["auc"]:.3f}, 95% CI: {auc_lower:.3f}-{auc_upper:.3f})')
    
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Classifier')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold', labelpad=10)
    plt.title('ROC Curve with 95% Confidence Interval', fontsize=14, fontweight='bold', pad=15)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc='lower right', fontsize=11, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f" ROC curve saved to: {save_path}")
    
    plt.show()


# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("HYBRID SMORF PREDICTION MODEL (n_estimators = 500)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ------------------------------
    # 1. Load Data
    # ------------------------------
    print("\n Loading data...")
    data = pd.read_csv(Config.DATA_PATH, sep='\t')
    print(f"  Loaded {len(data)} sequences")
    print(f"  Columns: {data.columns.tolist()}")
    
    # ------------------------------
    # 2. Extract Features
    # ------------------------------
    print("\n[2/6] Extracting k-mer features...")
    sequences = data['sequence'].tolist()
    labels = data['class'].values
    
    # Convert to k-mer strings
    kmer_strings = [' '.join(get_kmers(seq)) for seq in sequences]
    
    # Vectorize
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
    X = vectorizer.fit_transform(kmer_strings)
    y = labels
    
    print(f"  ✓ Feature matrix shape: {X.shape}")
    print(f"  ✓ Class distribution - Positive: {sum(y==1)}, Negative: {sum(y==0)}")
    
    # ------------------------------
    # 3. Train-Test Split
    # ------------------------------
    print("\n Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    print(f" Training samples: {X_train.shape[0]}")
    print(f" Test samples: {X_test.shape[0]}")
    
    # ------------------------------
    # 4. Train Model (n_estimators = 500)
    # ------------------------------
    print("\n[4/6] Training hybrid ensemble model (n_estimators = 500)...")
    model = build_hybrid_model()
    model.fit(X_train, y_train)
    print(f"  ✓ Model training completed")
    
    # ------------------------------
    # 5. Evaluate Model
    # ------------------------------
    print("\n[5/6] Evaluating model...")
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    
    n_pos = sum(y_test == 1)
    n_neg = sum(y_test == 0)
    print_results(metrics, len(y_test), n_pos, n_neg)
    
    # ------------------------------
    # 6. Save Results 
    # ------------------------------
    print("\n Saving results...")
    
    # Create results directory if needed
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Save ROC curve
    plot_roc_curve(metrics, save_path=os.path.join(Config.RESULTS_DIR, 'roc_curve.png'))
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([{
        'accuracy': metrics['accuracy'],
        'accuracy_ci_lower': metrics['accuracy_ci'][0],
        'accuracy_ci_upper': metrics['accuracy_ci'][1],
        'precision': metrics['precision'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'f1_score': metrics['f1_score'],
        'mcc': metrics['mcc'],
        'mcc_ci_lower': metrics['mcc_ci'][0],
        'mcc_ci_upper': metrics['mcc_ci'][1],
        'auc': metrics['auc'],
        'auc_ci_lower': metrics['auc_ci'][0],
        'auc_ci_upper': metrics['auc_ci'][1]
    }])
    metrics_df.to_csv(os.path.join(Config.RESULTS_DIR, 'metrics.csv'), index=False)
    print(f"  ✓ Metrics saved to: {Config.RESULTS_DIR}/metrics.csv")
    
    # Optional: Save model (commented by default - uncomment if needed)
    # save_model_package(model, vectorizer)
    
    # ------------------------------
    # Summary
    # ------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Random Forest n_estimators: {Config.RF_N_ESTIMATORS}")
    print(f"  Gradient Boosting n_estimators: {Config.GB_N_ESTIMATORS}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} (95% CI: {metrics['accuracy_ci'][0]:.4f}-{metrics['accuracy_ci'][1]:.4f})")
    print(f"  AUC: {metrics['auc']:.4f} (95% CI: {metrics['auc_ci'][0]:.4f}-{metrics['auc_ci'][1]:.4f})")
    print(f"  MCC: {metrics['mcc']:.4f} (95% CI: {metrics['mcc_ci'][0]:.4f}-{metrics['mcc_ci'][1]:.4f})")
    print(f"\n✓ Results saved to: {Config.RESULTS_DIR}/")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return model, vectorizer, metrics


# ============================================
# PREDICTION FUNCTION FOR NEW SEQUENCES
# ============================================

def predict_smorfs(sequences, model_path=Config.MODEL_SAVE_PATH):
    """
    Predict smORFs for new sequences using a trained model.
    
    Parameters
    ----------
    sequences : list
        List of DNA sequences
    model_path : str
        Path to the saved model package
    
    Returns
    -------
    tuple : (predictions, probabilities)
    """
    # Load model and vectorizer
    model, vectorizer = load_model_package(model_path)
    
    # Extract features
    kmer_strings = [' '.join(get_kmers(seq)) for seq in sequences]
    X = vectorizer.transform(kmer_strings)
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities



# SCRIPT ENTRY POINT

if __name__ == "__main__":
    model, vectorizer, metrics = main()