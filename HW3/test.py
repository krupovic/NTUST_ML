import argparse
import json
import os
import torch
import torch.nn as nn
import numpy as np
from models.mlp import MLP


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved checkpoint on test set")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pth file (required unless using --xgb)")
    p.add_argument("--dataset", choices=["mnist", "adult", "synthetic"], default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=str, default=None, help="Optional CSV path to save predictions")
    p.add_argument("--plot-dir", type=str, default=None, help="Optional directory to save evaluation plots (confusion matrix, ROC)")
    p.add_argument("--xgb", type=str, default=None, help="Optional path to a pickled XGBoost/SVM model to evaluate instead of a PyTorch checkpoint")
    p.add_argument("--threshold", type=float, default=None, help="Optional decision threshold for binary outputs (overrides 0.5)")
    
    args = p.parse_args()
    
    # Validate that either --ckpt or --xgb is provided
    if args.ckpt is None and args.xgb is None:
        p.error("Either --ckpt or --xgb must be provided")
    
    return args


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt.get('meta', None)
    return ckpt, meta


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Check if we should use XGBoost/SVM model instead of PyTorch checkpoint
    if args.xgb is not None:
        # Load XGBoost/SVM model
        import pickle
        with open(args.xgb, 'rb') as f:
            xgb_model = pickle.load(f)
        print(f"Loaded XGBoost/SVM model from {args.xgb}")
        
        # Try to load metadata from results file for preprocessing options
        model_dir = os.path.dirname(args.xgb)
        results_files = ['svm_results.json', 'xgb_run_meta.json']
        meta = None
        
        for results_file in results_files:
            results_path = os.path.join(model_dir, results_file)
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        results_data = json.load(f)
                        meta = results_data.get('metadata', {})
                        break
                except Exception as e:
                    print(f"Warning: Could not load metadata from {results_path}: {e}")
                    continue
        
        ckpt = None
    else:
        ckpt, meta = load_checkpoint(args.ckpt, device)

    # If meta exists, prefer its dataset unless user overrides
    dataset = args.dataset if args.dataset is not None else (meta.get('dataset') if meta else None)
    # label_classes is populated only for some datasets (adult). Ensure it's defined
    # so later code can safely check it for any dataset.
    label_classes = None
    if dataset is None:
        raise ValueError('Dataset must be specified either in checkpoint meta or via --dataset')

    # load data
    if dataset == 'mnist':
        from data.mnist_loader import get_mnist_loaders

        train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
        input_dim = 28 * 28
        num_classes = 10
    elif dataset == 'adult':
        from data.adult_loader import get_adult_loaders

        # Apply preprocessing options from metadata if available
        loader_kwargs = {'batch_size': args.batch_size}
        if meta and 'preprocessing' in meta:
            preprocessing = meta['preprocessing']
            if preprocessing.get('one_hot_only'):
                loader_kwargs['one_hot_only'] = True
            if preprocessing.get('feature_selection', 'none') != 'none':
                loader_kwargs['feature_selection'] = preprocessing['feature_selection']
                loader_kwargs['feature_threshold'] = preprocessing.get('feature_threshold', 0.02)
        
        train_loader, val_loader, test_loader = get_adult_loaders(**loader_kwargs)
        xb, yb = next(iter(test_loader))
        input_dim = xb.shape[1]
        num_classes = int(yb.max().item()) + 1
        # try to load preprocessing artifact to recover label class names
        label_classes = None
        try:
            preproc_path = None
            if meta and 'preproc' in meta:
                preproc_path = meta['preproc']
            else:
                candidate = os.path.join(os.path.dirname(__file__), 'data', 'preproc_adult.pkl')
                if os.path.exists(candidate):
                    preproc_path = candidate
            if preproc_path is not None and os.path.exists(preproc_path):
                import pickle

                with open(preproc_path, 'rb') as f:
                    preproc = pickle.load(f)
                    label_classes = preproc.get('label_classes')
        except Exception:
            label_classes = None
    else:
        # synthetic
        from torch.utils.data import TensorDataset, DataLoader

        input_dim = meta['input_dim'] if meta else 32
        num_classes = meta['num_classes'] if meta else 3
        X = torch.randn(1024, input_dim)
        y = torch.randint(0, num_classes, (1024,))
        test_loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)

    # construct model from meta when available - skip if using XGBoost
    if args.xgb is None:
        if meta is not None:
            # Check if this is a CNN checkpoint (has input_shape) or MLP checkpoint (has input_dim)
            if 'input_shape' in meta:
                # CNN checkpoint
                from models.cnn import CNN
                model = CNN(
                    input_shape=meta['input_shape'],
                    num_classes=meta['num_classes'],
                    num_conv_layers=meta['num_conv_layers'],
                    base_channels=meta['base_channels'],
                    channel_multiplier=meta['channel_multiplier'],
                    kernel_size=meta['kernel_size'],
                    pool_size=meta['pool_size'],
                    fc_sizes=meta['fc_sizes'],
                    dropout=meta['dropout']
                )
            elif 'input_dim' in meta:
                # MLP checkpoint
                activation_map = {
                    'relu': nn.ReLU,
                    'tanh': nn.Tanh,
                    'sigmoid': nn.Sigmoid,
                    'leakyrelu': nn.LeakyReLU,
                }
                act_cls = activation_map.get(meta.get('activation', 'relu'), nn.ReLU)
                if meta.get('loss') == 'bce_logits':
                    out_dim = 1
                else:
                    out_dim = meta.get('num_classes')
                model = MLP(
                    input_dim=meta['input_dim'],
                    hidden_sizes=meta['hidden_sizes'],
                    output_dim=out_dim,
                    activation=act_cls,
                    dropout=float(meta.get('dropout', 0.0)),
                )
            else:
                # Fallback - assume MLP
                model = MLP(input_dim=input_dim, hidden_sizes=[128, 64], output_dim=num_classes)
        else:
            model = MLP(input_dim=input_dim, hidden_sizes=[128, 64], output_dim=num_classes)

        model.load_state_dict(ckpt['model_state'])
        model.to(device)
        model.eval()
    else:
        model = None  # Using XGBoost instead

    import sklearn.metrics as skm

    all_targets = []
    all_preds = []
    all_probs = []

    if args.xgb is not None:
        # Use XGBoost model for prediction
        print("Running XGBoost evaluation...")
        all_X = []
        for xb, yb in test_loader:
            xb_np = xb.numpy()
            # Flatten image data for XGBoost (MNIST images need flattening)
            if len(xb_np.shape) > 2:  # Image data (batch, channels, height, width)
                xb_np = xb_np.reshape(xb_np.shape[0], -1)
            all_X.append(xb_np)
            all_targets.extend(yb.numpy().tolist())
        
        X_test = np.vstack(all_X)
        
        # Get XGBoost predictions
        if hasattr(xgb_model, 'predict_proba'):
            # Classifier with probability output
            probs = xgb_model.predict_proba(X_test)
            if probs.shape[1] == 2:
                # Binary classification
                probs_pos = probs[:, 1]
                threshold = args.threshold if args.threshold is not None else 0.5
                preds = (probs_pos >= threshold).astype(int)
                probs_all = [[1.0 - p, float(p)] for p in probs_pos]
            else:
                # Multiclass
                preds = probs.argmax(axis=1)
                probs_all = probs.tolist()
        else:
            # Regressor or classifier without predict_proba
            raw_preds = xgb_model.predict(X_test)
            if len(np.unique(all_targets)) == 2:
                # Binary classification
                threshold = args.threshold if args.threshold is not None else 0.5
                preds = (raw_preds >= threshold).astype(int)
                probs_all = [[1.0 - p, float(p)] for p in raw_preds]
            else:
                # Assume regression or integer predictions
                preds = raw_preds.round().astype(int)
                probs_all = [[float(p)] for p in raw_preds]
        
        all_preds = preds.tolist()
        all_probs = probs_all
    else:
        # Use PyTorch model for prediction
        print("Running PyTorch model evaluation...")
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                # handle binary vs multiclass outputs
                if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                    # binary output (single logit)
                    probs_pos = torch.sigmoid(logits.view(-1)).cpu().numpy()
                    threshold = args.threshold if args.threshold is not None else 0.5
                    preds = (probs_pos >= threshold).astype(int)
                    # store per-class probs as [prob0, prob1]
                    probs_all = [[1.0 - p, float(p)] for p in probs_pos]
                    all_targets.extend(yb.numpy().tolist())
                    all_preds.extend(preds.tolist())
                    all_probs.extend(probs_all)
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    all_targets.extend(yb.numpy().tolist())
                    all_preds.extend(preds.tolist())
                    all_probs.extend(probs.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    print('Accuracy:', skm.accuracy_score(y_true, y_pred))
    if label_classes is not None:
        try:
            print('Classification report:\n', skm.classification_report(y_true, y_pred, target_names=label_classes))
        except Exception:
            print('Classification report:\n', skm.classification_report(y_true, y_pred))
    else:
        print('Classification report:\n', skm.classification_report(y_true, y_pred))
    print('Confusion matrix:\n', skm.confusion_matrix(y_true, y_pred))

    # ROC AUC for binary
    if len(np.unique(y_true)) == 2:
        try:
            # take probability of class 1
            probs_pos = np.array(all_probs)[:, 1]
            print('ROC AUC:', skm.roc_auc_score(y_true, probs_pos))
        except Exception:
            pass

    # optional CSV output
    if args.out:
        import csv

        with open(args.out, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['index', 'true', 'pred'] + [f'prob_{i}' for i in range(len(all_probs[0]))]
            writer.writerow(header)
            for i, (t, p, probs) in enumerate(zip(all_targets, all_preds, all_probs)):
                t_out = label_classes[t] if (label_classes is not None and int(t) < len(label_classes)) else t
                p_out = label_classes[p] if (label_classes is not None and int(p) < len(label_classes)) else p
                writer.writerow([i, t_out, p_out] + probs)
        print(f'Wrote predictions to {args.out}')

    # plotting (confusion matrix, ROC if available)
    plot_dir = args.plot_dir if args.plot_dir is not None else os.path.dirname(args.ckpt)
    os.makedirs(plot_dir, exist_ok=True)

    try:
        # confusion matrix plot
        cm = skm.confusion_matrix(y_true, y_pred)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion matrix')
        # tick labels
        labels = np.unique(np.concatenate([y_true, y_pred]))
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.tight_layout()
        cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
        fig.savefig(cm_path)
        plt.close(fig)
        print(f'Saved confusion matrix to {cm_path}')

        # ROC curve for binary or multiclass (one-vs-rest)
        if len(np.unique(y_true)) == 2:
            probs_pos = np.array(all_probs)[:, 1]
            fpr, tpr, _ = skm.roc_curve(y_true, probs_pos)
            auc = skm.roc_auc_score(y_true, probs_pos)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC curve')
            ax.legend()
            roc_path = os.path.join(plot_dir, 'roc_curve.png')
            fig.savefig(roc_path)
            plt.close(fig)
            print(f'Saved ROC curve to {roc_path}')
        else:
            # multiclass: plot per-class ROC curves (one-vs-rest)
            try:
                from sklearn.preprocessing import label_binarize

                classes = np.unique(y_true)
                Y = label_binarize(y_true, classes=classes)
                probs_arr = np.array(all_probs)
                fig, ax = plt.subplots(figsize=(8, 6))
                for i, c in enumerate(classes):
                    if probs_arr.shape[1] == len(classes):
                        scores = probs_arr[:, i]
                    else:
                        continue
                    fpr, tpr, _ = skm.roc_curve(Y[:, i], scores)
                    auc = skm.roc_auc_score(Y[:, i], scores)
                    ax.plot(fpr, tpr, label=f'class {c} (AUC={auc:.3f})')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Multiclass ROC (one-vs-rest)')
                ax.legend()
                roc_multi_path = os.path.join(plot_dir, 'roc_multiclass.png')
                fig.savefig(roc_multi_path)
                plt.close(fig)
                print(f'Saved multiclass ROC plot to {roc_multi_path}')
            except Exception:
                pass
    except Exception as e:
        print(f'Failed to create plots: {e}')


if __name__ == '__main__':
    main()
