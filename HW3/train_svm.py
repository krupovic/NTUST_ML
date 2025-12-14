#!/usr/bin/env python3
"""
SVM Training Script
Supports both binary (Adult) and multi-class (MNIST) classification with comprehensive evaluation.
"""

import argparse
import json
import os
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Train SVM on specified dataset')
    
    # Dataset and I/O
    parser.add_argument('--dataset', choices=['mnist', 'adult'], required=True,
                       help='Dataset to train on')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for data loading')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Validation split size (for adult dataset)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit number of training samples (for quick testing)')
    
    # SVM Hyperparameters
    parser.add_argument('--kernel', choices=['linear', 'poly', 'rbf', 'sigmoid'], default='rbf',
                       help='SVM kernel function')
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter')
    parser.add_argument('--gamma', choices=['scale', 'auto'], default='scale',
                       help='Kernel coefficient (for rbf, poly, sigmoid)')
    parser.add_argument('--degree', type=int, default=3,
                       help='Degree for polynomial kernel')
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='none',
                       help='Class weight strategy')
    
    # Hyperparameter tuning
    parser.add_argument('--grid-search', action='store_true',
                       help='Perform grid search for hyperparameter tuning')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of cross-validation folds for grid search')
    
    # Multi-class strategy
    parser.add_argument('--multiclass', choices=['ovr', 'ovo'], default='ovr',
                       help='Multi-class strategy: one-vs-rest or one-vs-one')
    
    # Output and evaluation
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save model and results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Preprocessing options (for Adult dataset)
    parser.add_argument('--one-hot-only', action='store_true',
                       help='Use only one-hot encoding for Adult dataset')
    parser.add_argument('--feature-selection', choices=['pearson', 'none'], default='none',
                       help='Feature selection method')
    parser.add_argument('--feature-threshold', type=float, default=0.02,
                       help='Feature selection threshold')
    
    return parser.parse_args()


def load_dataset(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset and return train/val/test splits"""
    if args.dataset == 'mnist':
        from data.mnist_loader import get_mnist_loaders
        train_loader, val_loader, test_loader = get_mnist_loaders(
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
    elif args.dataset == 'adult':
        from data.adult_loader import get_adult_loaders
        
        # Configure preprocessing options
        loader_kwargs = {
            'batch_size': args.batch_size,
            'val_size': args.val_size,
            'max_samples': args.max_samples,
            'one_hot_only': args.one_hot_only,
        }
        
        if args.feature_selection != 'none':
            loader_kwargs['feature_selection'] = args.feature_selection
            loader_kwargs['feature_threshold'] = args.feature_threshold
        
        train_loader, val_loader, test_loader = get_adult_loaders(**loader_kwargs)
    
    # Convert PyTorch DataLoaders to numpy arrays
    def loader_to_arrays(loader, flatten_images=False):
        X_parts, y_parts = [], []
        for batch_x, batch_y in loader:
            batch_x_np = batch_x.numpy()
            if flatten_images and len(batch_x_np.shape) > 2:
                # Flatten image data: (batch, channels, height, width) -> (batch, features)
                batch_x_np = batch_x_np.reshape(batch_x_np.shape[0], -1)
            X_parts.append(batch_x_np)
            y_parts.append(batch_y.numpy())
        return np.vstack(X_parts), np.concatenate(y_parts)
    
    # For MNIST, we need to flatten image data for SVM
    flatten_images = (args.dataset == 'mnist')
    X_train, y_train = loader_to_arrays(train_loader, flatten_images=flatten_images)
    X_val, y_val = loader_to_arrays(val_loader, flatten_images=flatten_images)
    X_test, y_test = loader_to_arrays(test_loader, flatten_images=flatten_images)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_grid_search_params(kernel: str, dataset: str) -> dict:
    """Get grid search parameters based on kernel and dataset"""
    base_params = {}
    
    if kernel == 'rbf':
        if dataset == 'mnist':
            base_params = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        else:  # adult
            base_params = {
                'C': [0.01, 0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
    elif kernel == 'linear':
        base_params = {
            'C': [0.01, 0.1, 1, 10, 100]
        }
    elif kernel == 'poly':
        base_params = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
    elif kernel == 'sigmoid':
        base_params = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
    
    return base_params


def train_svm(X_train: np.ndarray, y_train: np.ndarray, args) -> SVC:
    """Train SVM with optional grid search"""
    
    # Determine class weight
    class_weight = 'balanced' if args.class_weight == 'balanced' else None
    
    if args.grid_search:
        print("Performing grid search for hyperparameter tuning...")
        
        # Create base SVM
        svm = SVC(
            kernel=args.kernel,
            degree=args.degree,
            class_weight=class_weight,
            decision_function_shape=args.multiclass,
            random_state=args.seed,
            probability=True  # Enable probability estimates for ROC/PR curves
        )
        
        # Get parameter grid
        param_grid = get_grid_search_params(args.kernel, args.dataset)
        
        # Perform grid search
        grid_search = GridSearchCV(
            svm, param_grid, 
            cv=args.cv_folds, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 if args.verbose else 0
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        print(f"Grid search completed in {search_time:.1f}s")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    else:
        print("Training SVM with specified parameters...")
        
        # Create SVM with specified parameters
        svm = SVC(
            kernel=args.kernel,
            C=args.C,
            gamma=args.gamma,
            degree=args.degree,
            class_weight=class_weight,
            decision_function_shape=args.multiclass,
            random_state=args.seed,
            probability=True
        )
        
        start_time = time.time()
        svm.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"Training completed in {train_time:.1f}s")
        
        return svm, None, None


def evaluate_svm(model: SVC, X_test: np.ndarray, y_test: np.ndarray, 
                 class_names: Optional[list] = None) -> dict:
    """Comprehensive SVM evaluation"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    if class_names is not None:
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    else:
        class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_proba.tolist(),
        'support_vectors': model.n_support_.tolist() if hasattr(model, 'n_support_') else None
    }
    
    # ROC AUC for binary classification
    unique_classes = np.unique(y_test)
    if len(unique_classes) == 2:
        # Binary classification
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        avg_precision = average_precision_score(y_test, y_proba[:, 1])
        
        results.update({
            'roc_auc': roc_auc,
            'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'pr_curve': {'precision': precision.tolist(), 'recall': recall.tolist()},
            'average_precision': avg_precision
        })
        
    else:
        # Multi-class classification - one-vs-rest ROC
        try:
            y_test_binarized = label_binarize(y_test, classes=unique_classes)
            roc_auc = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr', average='macro')
            results['roc_auc_macro'] = roc_auc
        except Exception:
            pass
    
    return results


def plot_results(results: dict, args, save_dir: str):
    """Generate and save evaluation plots"""
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = np.array(results['confusion_matrix'])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - SVM ({args.dataset.upper()})')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}',
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC Curve (binary classification)
    if 'roc_curve' in results:
        plt.figure(figsize=(8, 6))
        plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'], 
                label=f"ROC Curve (AUC = {results['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - SVM ({args.dataset.upper()})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(results['pr_curve']['recall'], results['pr_curve']['precision'],
                label=f"PR Curve (AP = {results['average_precision']:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - SVM ({args.dataset.upper()})')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()


def save_results(model: SVC, results: dict, args, best_params=None, best_score=None):
    """Save model and results"""
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.checkpoint_dir, 'svm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved SVM model to {model_path}")
    
    # Save results
    results_path = os.path.join(args.checkpoint_dir, 'svm_results.json')
    
    # Prepare metadata
    metadata = {
        'dataset': args.dataset,
        'kernel': args.kernel,
        'C': args.C if not args.grid_search else best_params.get('C', args.C),
        'gamma': args.gamma if not args.grid_search else best_params.get('gamma', args.gamma),
        'degree': args.degree if not args.grid_search else best_params.get('degree', args.degree),
        'class_weight': args.class_weight,
        'multiclass_strategy': args.multiclass,
        'grid_search': args.grid_search,
        'best_params': best_params,
        'best_cv_score': best_score,
        'preprocessing': {
            'one_hot_only': args.one_hot_only,
            'feature_selection': args.feature_selection,
            'feature_threshold': args.feature_threshold
        }
    }
    
    # Combine results with metadata
    full_results = {**results, 'metadata': metadata}
    
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    # Generate plots
    plot_results(results, args, args.checkpoint_dir)
    print(f"Saved plots to {args.checkpoint_dir}")


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"Training SVM on {args.dataset.upper()} dataset")
    print(f"Kernel: {args.kernel}, C: {args.C}, Gamma: {args.gamma}")
    if args.grid_search:
        print(f"Grid search enabled with {args.cv_folds}-fold CV")
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Train SVM
    print(f"\nTraining SVM...")
    train_result = train_svm(X_train, y_train, args)
    
    # Unpack results properly
    model, best_params, best_score = train_result
    
    # Print model info
    print(f"\nSVM Info:")
    print(f"Support vectors: {model.n_support_}")
    print(f"Total support vectors: {model.support_vectors_.shape[0]}")
    print(f"Kernel: {model.kernel}")
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    class_names = ['<=50K', '>50K'] if args.dataset == 'adult' else None
    results = evaluate_svm(model, X_test, y_test, class_names)
    
    # Print results
    print(f"\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    if 'roc_auc' in results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"Average Precision: {results['average_precision']:.4f}")
    elif 'roc_auc_macro' in results:
        print(f"ROC AUC (macro): {results['roc_auc_macro']:.4f}")
    
    # Print classification report
    print(f"\nClassification Report:")
    if class_names:
        from sklearn.metrics import classification_report
        print(classification_report(y_test, results['predictions'], target_names=class_names))
    else:
        from sklearn.metrics import classification_report
        print(classification_report(y_test, results['predictions']))
    
    # Save everything
    save_results(model, results, args, best_params, best_score)
    
    print(f"\nTraining completed! Results saved to {args.checkpoint_dir}")


if __name__ == '__main__':
    main()