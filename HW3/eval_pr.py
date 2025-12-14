import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Compute precision-recall curve and suggest thresholds")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
    p.add_argument("--dataset", choices=["mnist", "adult", "synthetic"], default=None)
    p.add_argument("--split", choices=["val", "test"], default="val", help="Which split to evaluate")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--plot-dir", default=None)
    p.add_argument("--out-csv", default=None, help="Optional CSV to write threshold/precision/recall/f1 rows")
    p.add_argument("--target-precisions", nargs='*', type=float, default=[0.8, 0.9], help="List of precision targets to report thresholds for")
    return p.parse_args()


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt.get('meta', None)
    return ckpt, meta


def build_model_from_meta(meta):
    from models.mlp import MLP
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
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt, meta = load_ckpt(args.ckpt, device)

    # dataset selection
    dataset = args.dataset if args.dataset is not None else (meta.get('dataset') if meta else None)
    if dataset is None:
        raise ValueError('Dataset must be specified either in checkpoint meta or via --dataset')

    label_classes = None

    # load data
    if dataset == 'mnist':
        from data.mnist_loader import get_mnist_loaders

        train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
        loader = val_loader if args.split == 'val' else test_loader
    elif dataset == 'adult':
        from data.adult_loader import get_adult_loaders

        train_loader, val_loader, test_loader = get_adult_loaders(batch_size=args.batch_size)
        loader = val_loader if args.split == 'val' else test_loader
        # try to load label names from preprocessing artifact
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
        loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=False)

    # build model
    if meta is not None:
        model = build_model_from_meta(meta)
    else:
        # fallback generic
        from models.mlp import MLP

        xb, yb = next(iter(loader))
        input_dim = xb.shape[1]
        num_classes = int(yb.max().item()) + 1
        model = MLP(input_dim=input_dim, hidden_sizes=[128, 64], output_dim=num_classes)

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                probs_pos = torch.sigmoid(logits.view(-1)).cpu().numpy()
                probs = np.vstack([1.0 - probs_pos, probs_pos]).T
                all_probs.append(probs)
                all_targets.extend(yb.numpy().tolist())
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_targets.extend(yb.numpy().tolist())

    all_probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_targets)

    import sklearn.metrics as skm
    import matplotlib.pyplot as plt

    plot_dir = args.plot_dir if args.plot_dir is not None else os.path.dirname(args.ckpt)
    os.makedirs(plot_dir, exist_ok=True)

    out_csv = args.out_csv if args.out_csv is not None else os.path.join(plot_dir, 'pr_thresholds.csv')

    rows = []

    # binary case
    if all_probs.shape[1] == 2 and len(np.unique(y_true)) == 2:
        probs_pos = all_probs[:, 1]
        precision, recall, thresholds = skm.precision_recall_curve(y_true, probs_pos)
        ap = skm.average_precision_score(y_true, probs_pos)

        # precision_recall_curve returns arrays where thresholds length = len(precision)-1
        threshs = np.append(thresholds, 1.0)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-12)

        # find best F1 (ignore last point which corresponds to threshold > max)
        best_idx = np.nanargmax(f1s)
        best_thresh = threshs[best_idx]
        best_prec = precision[best_idx]
        best_rec = recall[best_idx]
        best_f1 = f1s[best_idx]

        print(f'Average precision (AP): {ap:.4f}')
        print(f'Best F1={best_f1:.4f} at threshold={best_thresh:.4f} (precision={best_prec:.4f}, recall={best_rec:.4f})')

        # thresholds achieving target precisions
        for target in args.target_precisions:
            # find first threshold where precision >= target
            inds = np.where(precision >= target)[0]
            if inds.size > 0:
                idx = inds[0]
                thr = threshs[idx]
                p = precision[idx]
                r = recall[idx]
                f1 = f1s[idx]
                print(f'Precision >= {target:.2f} at threshold {thr:.4f}: prec={p:.4f}, rec={r:.4f}, f1={f1:.4f}')
            else:
                print(f'No threshold achieves precision >= {target:.2f}')

        # save CSV of thresholds
        import csv

        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['threshold', 'precision', 'recall', 'f1'])
            for t, p, r, f1 in zip(threshs, precision, recall, f1s):
                writer.writerow([t, p, r, f1])
        print(f'Wrote thresholds to {out_csv}')

        # plot PR curve
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, label=f'AP={ap:.4f}')
        ax.scatter([best_rec], [best_prec], color='red', label=f'Best F1={best_f1:.3f}\nthr={best_thresh:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        pr_path = os.path.join(plot_dir, 'pr_curve.png')
        fig.savefig(pr_path)
        plt.close(fig)
        print(f'Saved PR curve to {pr_path}')

    else:
        # multiclass: one-vs-rest PR curves
        from sklearn.preprocessing import label_binarize

        classes = np.unique(y_true)
        Y = label_binarize(y_true, classes=classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, c in enumerate(classes):
            if all_probs.shape[1] != len(classes):
                print(f'Skipping class {c} because prob shape {all_probs.shape} != num classes {len(classes)}')
                continue
            scores = all_probs[:, i]
            precision, recall, thresholds = skm.precision_recall_curve(Y[:, i], scores)
            ap = skm.average_precision_score(Y[:, i], scores)
            ax.plot(recall, precision, label=f'class {c} (AP={ap:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Multiclass Precision-Recall (one-vs-rest)')
        ax.legend()
        pr_path = os.path.join(plot_dir, 'pr_curve_multiclass.png')
        fig.savefig(pr_path)
        plt.close(fig)
        print(f'Saved multiclass PR curves to {pr_path}')


if __name__ == '__main__':
    main()
