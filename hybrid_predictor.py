#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Machine Learning Model for Bacterial smORF Prediction
============================================================

This script implements a hybrid ensemble model combining Random Forest and 
Gradient Boosting classifiers to predict small open reading frames (smORFs) 
in bacterial genomes.

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
import math
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    auc
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
    MODEL_SAVE_PATH = 'hybrid_smorff_model.joblib'
    RESULTS_DIR = 'results'
    
    # Random Forest Hyperparameters
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = 'balanced'
    
    # Gradient Boosting Hyperparameters
    GB_N_ESTIMATORS = 500
    GB_LEARNING_RATE = 0.1
    GB_MAX_DEPTH = 10
    GB_SUBSAMPLE = 1.0
    GB_MIN_SAMPLES_SPLIT = 2
    GB_MIN_SAMPLES_LEAF = 1
    
    # Data split parameters
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
    """Convert DNA sequence into k-mer words."""
    sequence = str(sequence).upper().strip()
    if len(sequence) < size:
        return [sequence]
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]


# ============================================
# MODEL BUILDING
# ============================================

def build_hybrid_model(random_state=Config.RANDOM_STATE):
    """Build the hybrid ensemble model with full hyperparameters."""
    
    rf = RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        min_samples_split=Config.RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.RF_MIN_SAMPLES_LEAF,
        class_weight=Config.RF_CLASS_WEIGHT,
        random_state=random_state,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=Config.GB_N_ESTIMATORS,
        learning_rate=Config.GB_LEARNING_RATE,
        max_depth=Config.GB_MAX_DEPTH,
        subsample=Config.GB_SUBSAMPLE,
        min_samples_split=Config.GB_MIN_SAMPLES_SPLIT,
        min_samples_leaf=Config.GB_MIN_SAMPLES_LEAF,
        random_state=random_state
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    return ensemble


# ============================================
# MODEL SAVING AND LOADING
# ============================================

def save_model_package(model, vectorizer, filepath=Config.MODEL_SAVE_PATH):
    """
    Save model and vectorizer together in a single joblib file.
    """
    package = {
        'model': model,
        'vectorizer': vectorizer,
        'model_type': 'VotingClassifier (Random Forest + Gradient Boosting)',
        'version': '2.0',
        'kmer_size': Config.K_MER_SIZE,
        'features': '6-mer frequency'
    }
    
    joblib.dump(package, filepath)
    print("Model saved to: {0}".format(filepath))
    return filepath


def load_model_package(filepath=Config.MODEL_SAVE_PATH):
    """
    Load model and vectorizer from a single joblib file.
    """
    package = joblib.load(filepath)
    print("Model loaded from: {0}".format(filepath))
    return package['model'], package['vectorizer']


# ============================================
# CROSS-VALIDATION FUNCTION
# ============================================

def perform_cross_validation(
    X,
    y,
    cv_folds=Config.CV_FOLDS,
    random_state=Config.RANDOM_STATE
):
    """
    Perform stratified k-fold cross-validation using pooled
    cross-validated predictions.

    This implementation follows the same evaluation strategy
    reported in the manuscript.
    """
    
    # Dynamic fold adjustment for small datasets
    min_samples_per_class = np.min(np.bincount(y))
    cv_folds = min(cv_folds, min_samples_per_class)
    
    if cv_folds < 2:
        print("Warning: Only {0} samples in smallest class. Using 2 folds.".format(min_samples_per_class))
        cv_folds = 2
    
    print("Performing {0}-fold stratified cross-validation...".format(cv_folds))

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    # pooled out-of-fold probabilities
    y_cv_proba = cross_val_predict(
        build_hybrid_model(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    y_cv_pred = (y_cv_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y, y_cv_pred)
    precision = precision_score(y, y_cv_pred, average="weighted")
    sensitivity = recall_score(y, y_cv_pred, average="weighted")
    f1 = f1_score(y, y_cv_pred, average="weighted")
    mcc = matthews_corrcoef(y, y_cv_pred)

    cm = confusion_matrix(y, y_cv_pred)

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    auc_score = roc_auc_score(y, y_cv_proba)

    print("")
    print("========== 10-Fold Cross Validation ==========")
    print("Accuracy      : {0:.4f}".format(accuracy))
    print("Precision     : {0:.4f}".format(precision))
    print("Sensitivity   : {0:.4f}".format(sensitivity))
    print("Specificity   : {0:.4f}".format(specificity))
    print("F1-score      : {0:.4f}".format(f1))
    print("MCC           : {0:.4f}".format(mcc))

    print("")
    print("Confusion Matrix")
    print(cm)

    print("")
    print("AUC           : {0:.4f}".format(auc_score))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "mcc": mcc,
        "auc": auc_score,
        "confusion_matrix": cm,
        "probabilities": y_cv_proba,
        "predictions": y_cv_pred,
        "cv_folds": cv_folds
    }


# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_model(y_true, y_pred, y_proba):
    """Calculate all evaluation metrics with confidence intervals."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc_score = auc(fpr, tpr)
    
    n_pos = sum(y_true == 1)
    n_neg = sum(y_true == 0)
    n_total = len(y_true)
    
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
        'tpr': tpr,
        'n_total': n_total,
        'n_pos': n_pos,
        'n_neg': n_neg
    }
    
    return metrics


def print_results(metrics, label='Hold-out Test Set'):
    """Print evaluation results in a formatted table."""
    print("")
    print("="*70)
    print("HYBRID MODEL PERFORMANCE -- {0}".format(label))
    print("="*70)
    print("")
    print("Sample size: {0} sequences ({1} positive, {2} negative)".format(
        metrics['n_total'], metrics['n_pos'], metrics['n_neg']))
    print("")
    print("{0:<18} {1:<12} {2:<25} {3:<12}".format('Metric', 'Value', '95% CI', 'SE'))
    print("-"*70)
    
    acc_lower, acc_upper = metrics['accuracy_ci']
    print("{0:<18} {1:<12.4f} [{2:.4f} - {3:.4f}] {4:<12}".format(
        'Accuracy', metrics['accuracy'], acc_lower, acc_upper, 'N/A'))
    
    auc_lower, auc_upper = metrics['auc_ci']
    print("{0:<18} {1:<12.4f} [{2:.4f} - {3:.4f}] {4:<12.4f}".format(
        'AUC', metrics['auc'], auc_lower, auc_upper, metrics['auc_se']))
    
    mcc_lower, mcc_upper = metrics['mcc_ci']
    print("{0:<18} {1:<12.4f} [{2:.4f} - {3:.4f}] {4:<12}".format(
        'MCC', metrics['mcc'], mcc_lower, mcc_upper, 'N/A'))
    
    print("{0:<18} {1:<12.4f} {2:<25} {3:<12}".format(
        'Precision', metrics['precision'], 'N/A', 'N/A'))
    print("{0:<18} {1:<12.4f} {2:<25} {3:<12}".format(
        'Sensitivity', metrics['sensitivity'], 'N/A', 'N/A'))
    print("{0:<18} {1:<12.4f} {2:<25} {3:<12}".format(
        'Specificity', metrics['specificity'], 'N/A', 'N/A'))
    print("{0:<18} {1:<12.4f} {2:<25} {3:<12}".format(
        'F1 Score', metrics['f1_score'], 'N/A', 'N/A'))
    
    cm = metrics['confusion_matrix']
    print("")
    print("="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("  True Negatives (TN): {0}".format(cm[0,0]))
    print("  False Positives (FP): {0}".format(cm[0,1]))
    print("  False Negatives (FN): {0}".format(cm[1,0]))
    print("  True Positives (TP): {0}".format(cm[1,1]))


# ============================================
# VISUALIZATION
# ============================================

def plot_roc_curve(metrics, save_path=None):
    """Plot ROC curve with confidence interval."""
    plt.figure(figsize=(8, 8))
    
    auc_lower, auc_upper = metrics['auc_ci']
    
    plt.plot(metrics['fpr'], metrics['tpr'], color='#1f77b4', lw=2.5,
             label='Hybrid Model (AUC = {0:.3f}, 95% CI: {1:.3f}-{2:.3f})'.format(
                 metrics['auc'], auc_lower, auc_upper))
    
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print("ROC curve saved to: {0}".format(save_path))
    
    plt.show()


# ============================================
# MAIN PIPELINE
# ============================================

def main():
    """Main execution function."""
    print("="*70)
    print("HYBRID SMORF PREDICTION MODEL")
    print("="*70)
    print("Started: {0}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # ------------------------------
    # 1. Load Data
    # ------------------------------
    print("")
    print("Loading data...")
    data = pd.read_csv(Config.DATA_PATH, sep='\t')
    print("  Loaded {0} sequences".format(len(data)))
    print("  {0} unique sequences".format(len(data['sequence'].unique())))
    
    # ------------------------------
    # 2. Extract Features
    # ------------------------------
    print("")
    print("Extracting k-mer features...")
    sequences = data['sequence'].tolist()
    labels = data['class'].values
    
    kmer_strings = [' '.join(get_kmers(seq)) for seq in sequences]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
    X = vectorizer.fit_transform(kmer_strings)
    y = labels
    
    print("  Feature matrix shape: {0}".format(X.shape))
    print("  Class distribution - Positive: {0}, Negative: {1}".format(
        sum(y==1), sum(y==0)))
    
    # ------------------------------
    # 3. Train-Test Split
    # ------------------------------
    print("")
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    print("  Training samples: {0}".format(X_train.shape[0]))
    print("  Test samples: {0}".format(X_test.shape[0]))
    
    # ------------------------------
    # 4. Cross-Validation on Training Set
    # ------------------------------
    print("")
    print("Performing cross-validation...")
    cv_metrics = perform_cross_validation(X_train, y_train)
    
    # ------------------------------
    # 5. Train Final Model
    # ------------------------------
    print("")
    print("Training final model...")
    model = build_hybrid_model()
    model.fit(X_train, y_train)
    print("  Training completed")
    
    # ------------------------------
    # 6. Save Model
    # ------------------------------
    print("")
    print("Saving model...")
    save_model_package(model, vectorizer)
    
    # ------------------------------
    # 7. Evaluate on Hold-out Test Set
    # ------------------------------
    print("")
    print("Evaluating hold-out test set...")
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    holdout_metrics = evaluate_model(y_test, y_pred, y_proba)
    print_results(holdout_metrics, label='Hold-out Test Set')
    
    # ------------------------------
    # 8. Save Results
    # ------------------------------
    print("")
    print("Saving results...")
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # ROC curve
    plot_roc_curve(holdout_metrics, save_path=os.path.join(Config.RESULTS_DIR, 'roc_curve.png'))
    
    # ROC data
    roc_df = pd.DataFrame({
        "FPR": holdout_metrics["fpr"],
        "TPR": holdout_metrics["tpr"]
    })
    roc_df.to_csv(
        os.path.join(Config.RESULTS_DIR, "roc_curve_data.csv"),
        index=False
    )
    print("  ROC data saved")
    
    # Metrics CSV with confidence intervals
    metrics_df = pd.DataFrame([{
        'Evaluation': 'Cross-Validation',
        'Accuracy': cv_metrics['accuracy'],
        'Precision': cv_metrics['precision'],
        'Sensitivity': cv_metrics['sensitivity'],
        'Specificity': cv_metrics['specificity'],
        'F1_score': cv_metrics['f1_score'],
        'MCC': cv_metrics['mcc'],
        'AUC': cv_metrics['auc']
    }, {
        'Evaluation': 'Hold-out Test',
        'Accuracy': holdout_metrics['accuracy'],
        'Accuracy_CI_Lower': holdout_metrics['accuracy_ci'][0],
        'Accuracy_CI_Upper': holdout_metrics['accuracy_ci'][1],
        'Precision': holdout_metrics['precision'],
        'Sensitivity': holdout_metrics['sensitivity'],
        'Specificity': holdout_metrics['specificity'],
        'F1_score': holdout_metrics['f1_score'],
        'MCC': holdout_metrics['mcc'],
        'MCC_CI_Lower': holdout_metrics['mcc_ci'][0],
        'MCC_CI_Upper': holdout_metrics['mcc_ci'][1],
        'AUC': holdout_metrics['auc'],
        'AUC_CI_Lower': holdout_metrics['auc_ci'][0],
        'AUC_CI_Upper': holdout_metrics['auc_ci'][1]
    }])
    metrics_df.to_csv(os.path.join(Config.RESULTS_DIR, 'metrics_summary.csv'), index=False)
    print("  Metrics saved")
    
    # ------------------------------
    # Summary
    # ------------------------------
    print("")
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    print("")
    print("Cross-Validation ({0} folds)".format(cv_metrics['cv_folds']))
    print("  Accuracy    : {0:.4f}".format(cv_metrics['accuracy']))
    print("  Precision   : {0:.4f}".format(cv_metrics['precision']))
    print("  Sensitivity : {0:.4f}".format(cv_metrics['sensitivity']))
    print("  Specificity : {0:.4f}".format(cv_metrics['specificity']))
    print("  F1-score    : {0:.4f}".format(cv_metrics['f1_score']))
    print("  MCC         : {0:.4f}".format(cv_metrics['mcc']))
    print("  AUC         : {0:.4f}".format(cv_metrics['auc']))
    
    print("")
    print("Hold-out Test Set")
    print("  Accuracy    : {0:.4f} (95% CI: {1:.4f}-{2:.4f})".format(
        holdout_metrics['accuracy'], 
        holdout_metrics['accuracy_ci'][0], 
        holdout_metrics['accuracy_ci'][1]))
    print("  Precision   : {0:.4f}".format(holdout_metrics['precision']))
    print("  Sensitivity : {0:.4f}".format(holdout_metrics['sensitivity']))
    print("  Specificity : {0:.4f}".format(holdout_metrics['specificity']))
    print("  F1-score    : {0:.4f}".format(holdout_metrics['f1_score']))
    print("  MCC         : {0:.4f} (95% CI: {1:.4f}-{2:.4f})".format(
        holdout_metrics['mcc'],
        holdout_metrics['mcc_ci'][0],
        holdout_metrics['mcc_ci'][1]))
    print("  AUC         : {0:.4f} (95% CI: {1:.4f}-{2:.4f})".format(
        holdout_metrics['auc'],
        holdout_metrics['auc_ci'][0],
        holdout_metrics['auc_ci'][1]))
    
    print("")
    print("Model saved to: {0}".format(Config.MODEL_SAVE_PATH))
    print("Results saved to: {0}/".format(Config.RESULTS_DIR))
    print("")
    print("Completed: {0}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("="*70)
    
    return model, vectorizer, holdout_metrics, cv_metrics


# ============================================
# PREDICTION FUNCTION
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
    model, vectorizer = load_model_package(model_path)
    
    kmer_strings = [' '.join(get_kmers(seq)) for seq in sequences]
    X = vectorizer.transform(kmer_strings)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    return predictions, probabilities


# ============================================
# SCRIPT ENTRY POINT
# ============================================

if __name__ == "__main__":
    model, vectorizer, holdout_metrics, cv_metrics = main()
