"""Train and evaluate an XGBoost baseline on Adult or MNIST datasets.

This script reuses the existing data loaders so features match the MLP/CNN runs.
It trains an XGBoost classifier, applies early stopping on the validation split, and writes
model + evaluation artifacts under `checkpoints/`.

Usage examples:
    python3 train_xgboost.py --dataset adult --n-estimators 200 --max-depth 6 --batch-size 256
    python3 train_xgboost.py --dataset mnist --n-estimators 800 --max-depth 12 --batch-size 512

"""
import argparse
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['adult', 'mnist'], required=True, help='Dataset to train on')
    p.add_argument('--n-estimators', type=int, default=200)
    p.add_argument('--max-depth', type=int, default=6)
    p.add_argument('--learning-rate', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='cpu')
    p.add_argument('--plot-dir', default='./checkpoints')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def collate_loader(loader, flatten_images=False):
    # loader yields (xb, yb) as tensors; collect into numpy arrays
    import torch
    X_parts = []
    y_parts = []
    for xb, yb in loader:
        xb_np = xb.detach().cpu().numpy()
        if flatten_images and len(xb_np.shape) > 2:
            # Flatten image data: (batch, channels, height, width) -> (batch, features)
            xb_np = xb_np.reshape(xb_np.shape[0], -1)
        X_parts.append(xb_np)
        y_parts.append(yb.detach().cpu().numpy())
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


def main():
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    # load data (uses same preprocessing as train/test scripts)
    if args.dataset == 'adult':
        from data.adult_loader import get_adult_loaders
        train_loader, val_loader, test_loader = get_adult_loaders(batch_size=args.batch_size)
    elif args.dataset == 'mnist':
        from data.mnist_loader import get_mnist_loaders
        train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    # For MNIST, we need to flatten image data for XGBoost
    flatten_images = (args.dataset == 'mnist')
    X_train, y_train = collate_loader(train_loader, flatten_images=flatten_images)
    X_val, y_val = collate_loader(val_loader, flatten_images=flatten_images)
    X_test, y_test = collate_loader(test_loader, flatten_images=flatten_images)

    # compute scale_pos_weight for XGBoost to handle imbalance (binary classification only)
    scale_pos_weight = 1.0
    if args.dataset == 'adult':  # binary classification
        num_pos = int((y_train == 1).sum())
        num_neg = int((y_train == 0).sum())
        if num_pos > 0:
            scale_pos_weight = float(num_neg) / float(max(1, num_pos))
    # For multi-class (MNIST), scale_pos_weight is not applicable

    try:
        import xgboost as xgb
    except Exception:
        raise RuntimeError('xgboost is required to run this script. Install via pip install xgboost')

    # Configure XGBoost parameters based on dataset
    xgb_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'use_label_encoder': False,
        'random_state': args.seed,
        'verbosity': 1,
    }
    
    if args.dataset == 'adult':  # binary classification
        xgb_params.update({
            'eval_metric': 'logloss',
            'scale_pos_weight': scale_pos_weight,
        })
    else:  # multi-class classification (MNIST)
        xgb_params.update({
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
        })
    
    model = xgb.XGBClassifier(**xgb_params)

    eval_set = [(X_val, y_val)]
    # some xgboost versions' sklearn wrapper don't accept early_stopping_rounds as a kwarg
    # try with early stopping first, otherwise fall back to training without it
    try:
        model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=eval_set, verbose=False)
    except TypeError:
        print('Warning: XGBoost version does not accept early_stopping_rounds keyword in XGBClassifier.fit(); training without early stopping.')
        model.fit(X_train, y_train, eval_set=eval_set)

    # predictions
    y_prob = model.predict_proba(X_test)
    y_pred = y_prob.argmax(axis=1)

    # metrics
    import sklearn.metrics as skm

    acc = skm.accuracy_score(y_test, y_pred)
    report = skm.classification_report(y_test, y_pred, output_dict=True)
    cm = skm.confusion_matrix(y_test, y_pred)
    
    # Binary classification specific metrics
    if y_prob.shape[1] == 2:  # binary
        ap = skm.average_precision_score(y_test, y_prob[:, 1])
        roc = skm.roc_auc_score(y_test, y_prob[:, 1])
    else:  # multi-class
        # For multi-class, compute macro-averaged metrics
        # average_precision_score needs different handling for multi-class
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(y_prob.shape[1]))
        ap = skm.average_precision_score(y_test_bin, y_prob, average='macro')
        roc = skm.roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

    results = {
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'average_precision': float(ap),
        'roc_auc': float(roc),
        'dataset': args.dataset,
        'n_classes': int(y_prob.shape[1]),
    }
    
    if args.dataset == 'adult':
        results['scale_pos_weight'] = float(scale_pos_weight)

    # save model and metadata
    model_path = os.path.join(args.plot_dir, 'xgb_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    meta = {
        'model': 'xgboost',
        'dataset': args.dataset,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_classes': int(y_prob.shape[1]),
    }
    
    if args.dataset == 'adult':
        meta['scale_pos_weight'] = scale_pos_weight
    with open(os.path.join(args.plot_dir, 'xgb_run_meta.json'), 'w') as f:
        json.dump({**meta, 'results': results}, f, indent=2)

    # save predictions CSV
    import csv

    out_csv = os.path.join(args.plot_dir, 'xgb_preds.csv')
    header = ['index', 'true', 'pred'] + [f'prob_{i}' for i in range(y_prob.shape[1])]
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, (t, p, probs) in enumerate(zip(y_test.tolist(), y_pred.tolist(), y_prob.tolist())):
            w.writerow([i, int(t), int(p)] + probs)

    # plotting (binary classification only for PR/ROC curves)
    if y_prob.shape[1] == 2:
        # precision-recall curve
        precision, recall, thresholds = skm.precision_recall_curve(y_test, y_prob[:, 1])
        ap_val = skm.average_precision_score(y_test, y_prob[:, 1])
        plt.figure(figsize=(6, 6))
        plt.plot(recall, precision, label=f'AP={ap_val:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'XGBoost Precision-Recall ({args.dataset.upper()})')
        plt.legend()
        pr_path = os.path.join(args.plot_dir, 'xgb_pr_curve.png')
        plt.tight_layout()
        plt.savefig(pr_path)
        plt.close()

        # ROC curve
        fpr, tpr, _ = skm.roc_curve(y_test, y_prob[:, 1])
        auc = skm.roc_auc_score(y_test, y_prob[:, 1])
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'XGBoost ROC ({args.dataset.upper()})')
        plt.legend()
        roc_path = os.path.join(args.plot_dir, 'xgb_roc.png')
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
    
    # Confusion matrix (for both binary and multi-class)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'XGBoost Confusion Matrix ({args.dataset.upper()})')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks)
    plt.yticks(tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    cm_path = os.path.join(args.plot_dir, 'xgb_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # write metrics summary
    with open(os.path.join(args.plot_dir, 'xgb_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved model to', model_path)
    print('Saved predictions to', out_csv)
    print('Saved metrics to', os.path.join(args.plot_dir, 'xgb_metrics.json'))


if __name__ == '__main__':
    main()
