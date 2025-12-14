import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.mlp import MLP
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Shallow MLP trainer for MNIST/Adult or synthetic smoke test")
    p.add_argument("--dataset", choices=["mnist", "adult", "synthetic"], default="synthetic")
    p.add_argument("--input-dim", type=int, default=784, help="input dimension (for synthetic or override)")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 64], help="hidden layer sizes")
    p.add_argument("--activation", choices=["relu", "tanh", "sigmoid", "leakyrelu"], default="relu", help="activation function")
    p.add_argument("--loss", choices=["cross_entropy", "bce_logits"], default="cross_entropy", help="loss/output type")
    p.add_argument("--val-size", type=float, default=0.1, help="validation split size (fraction)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=None, help="limit dataset samples for quick tests")
    p.add_argument("--use-weighted-sampler", action='store_true', help="use a WeightedRandomSampler to balance classes during training")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="directory to save model checkpoints")
    p.add_argument("--record-gradients", action='store_true', help="record gradient norms and flow for analysis")
    return p.parse_args()


def compute_gradient_norm(model):
    """Compute L2 norm of gradients across all parameters"""
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    return (total_norm ** 0.5) if param_count > 0 else 0.0


def compute_parameter_update_norm(model, prev_params):
    """Compute L2 norm of parameter updates"""
    total_norm = 0.0
    param_count = 0
    for (name, p), prev_p in zip(model.named_parameters(), prev_params):
        if p.requires_grad:
            update_norm = (p.data - prev_p).norm(2)
            total_norm += update_norm.item() ** 2
            param_count += 1
    return (total_norm ** 0.5) if param_count > 0 else 0.0


def record_layer_gradients(model):
    """Record gradient norms for each layer"""
    layer_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            layer_grads[name] = p.grad.data.norm(2).item()
        else:
            layer_grads[name] = 0.0
    return layer_grads


def get_data_loaders(args):
    if args.dataset == "mnist":
        from data.mnist_loader import get_mnist_loaders

        train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size, val_split=args.val_size, max_samples=args.max_samples)
        input_dim = 28 * 28
        num_classes = 10
    elif args.dataset == "adult":
        from data.adult_loader import get_adult_loaders

        train_loader, val_loader, test_loader = get_adult_loaders(batch_size=args.batch_size, val_size=args.val_size, max_samples=args.max_samples)
        # infer input dim/num_classes from a batch
        xb, yb = next(iter(train_loader))
        input_dim = xb.shape[1]
        num_classes = int(yb.max().item()) + 1
    else:  # synthetic
        from torch.utils.data import TensorDataset, DataLoader

        input_dim = args.input_dim
        num_classes = args.num_classes
        X = torch.randn(1024, input_dim)
        y = torch.randint(0, num_classes, (1024,))
        ds = TensorDataset(X, y)
        # split synthetic into train/val/test
        total = len(ds)
        val_n = int(total * args.val_size)
        train_n = total - val_n
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim, num_classes


def train_epoch(model, loader, opt, criterion, device, record_gradients=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Gradient recording
    gradient_norms = []
    layer_gradients = defaultdict(list)
    parameter_updates = []
    
    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        # Store previous parameters for update norm calculation
        if record_gradients:
            prev_params = [p.data.clone() for p in model.parameters() if p.requires_grad]
        
        # handle binary (BCEWithLogitsLoss) vs multiclass (CrossEntropyLoss)
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            yb = yb.float().to(device)
            logits = model(xb).view(-1)
            loss = criterion(logits, yb)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == yb.long()).sum().item()
        else:
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
        
        opt.zero_grad()
        loss.backward()
        
        # Record gradients before optimizer step
        if record_gradients:
            grad_norm = compute_gradient_norm(model)
            gradient_norms.append(grad_norm)
            
            # Record layer-wise gradients every 10 batches to avoid overhead
            if batch_idx % 10 == 0:
                layer_grads = record_layer_gradients(model)
                for layer_name, grad_norm in layer_grads.items():
                    layer_gradients[layer_name].append(grad_norm)
        
        opt.step()
        
        # Record parameter updates
        if record_gradients:
            update_norm = compute_parameter_update_norm(model, prev_params)
            parameter_updates.append(update_norm)
        
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    
    epoch_stats = {
        'loss': total_loss / total,
        'accuracy': correct / total
    }
    
    if record_gradients:
        epoch_stats.update({
            'avg_gradient_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'max_gradient_norm': np.max(gradient_norms) if gradient_norms else 0.0,
            'min_gradient_norm': np.min(gradient_norms) if gradient_norms else 0.0,
            'avg_parameter_update_norm': np.mean(parameter_updates) if parameter_updates else 0.0,
            'layer_gradients': dict(layer_gradients)
        })
    
    return epoch_stats


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            # handle BCEWithLogitsLoss vs CrossEntropyLoss
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                yb = yb.float().to(device)
                logits = model(xb).view(-1)
                loss = criterion(logits, yb)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
            else:
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device(args.device)
    train_loader, val_loader, test_loader, input_dim, num_classes = get_data_loaders(args)

    print(f"Dataset: {args.dataset}, input_dim={input_dim}, num_classes={num_classes}")

    # map activation
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'leakyrelu': nn.LeakyReLU,
    }

    activation_cls = activation_map.get(args.activation, nn.ReLU)

    # determine output dimension + loss
    if args.loss == 'bce_logits':
        # binary classification -> single logit output
        if num_classes != 2:
            # allow user to force binary by mapping classes to binary is not implemented automatically
            print('Warning: BCEWithLogitsLoss selected but dataset has num_classes != 2. Proceeding with output_dim=1 and using label==1 as positive.')
        output_dim = 1
        criterion = nn.BCEWithLogitsLoss()
    else:
        output_dim = num_classes
        criterion = nn.CrossEntropyLoss()

    model = MLP(input_dim=input_dim, hidden_sizes=args.hidden_sizes, output_dim=output_dim, activation=activation_cls, dropout=args.dropout)
    model.to(device)

    # prepare run metadata (used in checkpoints) so we can record class-weight info
    import os
    meta = {
        'dataset': args.dataset,
        'input_dim': int(input_dim),
        'hidden_sizes': list(args.hidden_sizes),
        'num_classes': int(num_classes),
        'activation': args.activation,
        'loss': args.loss,
        'dropout': float(args.dropout),
    }
    # attach path to preprocessor artifact if present (adult loader writes it)
    preproc_path = os.path.join(os.path.dirname(__file__), 'data', 'preproc_adult.pkl')
    if args.dataset == 'adult' and os.path.exists(preproc_path):
        meta['preproc'] = preproc_path

    # compute class weights from training data and optionally use weighted sampler
    train_dataset = train_loader.dataset
    try:
        # TensorDataset stores tensors in .tensors
        labels_arr = train_dataset.tensors[1].cpu().numpy()
    except Exception:
        # fallback: iterate dataset
        labels_list = []
        for _, y in train_dataset:
            if isinstance(y, torch.Tensor):
                labels_list.append(int(y.item()))
            else:
                labels_list.append(int(y))
        labels_arr = np.array(labels_list)

    class_counts = np.bincount(labels_arr, minlength=num_classes)
    total = labels_arr.shape[0]
    # avoid division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = total / (num_classes * class_counts)

    if args.loss == 'bce_logits' and num_classes == 2:
        # pos_weight for BCEWithLogitsLoss expects ratio of negative/positive
        pos = class_counts[1]
        neg = class_counts[0]
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        meta['class_weights'] = {'pos_weight': float((neg / pos))}
    else:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
        meta['class_weights'] = {'per_class': class_weights.tolist()}

    # optionally replace train_loader with a weighted sampler
    if args.use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        # sample weight for each sample is inverse to its class frequency
        sample_weights = 1.0 / class_counts[labels_arr]
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    opt = optim.Adam(model.parameters(), lr=args.lr)

    # checkpointing
    import json
    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = -1.0
    
    # Initialize gradient tracking
    gradient_history = defaultdict(list) if args.record_gradients else None

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, train_loader, opt, criterion, device, args.record_gradients)
        val_loss, val_acc = eval_epoch(model, val_loader if val_loader is not None else test_loader, criterion, device)
        
        train_loss = train_stats['loss']
        train_acc = train_stats['accuracy']
        
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        
        # Record gradient statistics
        if args.record_gradients:
            gradient_history['avg_gradient_norm'].append(train_stats['avg_gradient_norm'])
            gradient_history['max_gradient_norm'].append(train_stats['max_gradient_norm'])
            gradient_history['avg_parameter_update_norm'].append(train_stats['avg_parameter_update_norm'])
            print(f"  Gradient norm: avg={train_stats['avg_gradient_norm']:.6f}, max={train_stats['max_gradient_norm']:.6f}")
            print(f"  Parameter update norm: {train_stats['avg_parameter_update_norm']:.6f}")

        # record metrics
        if 'metrics' not in locals():
            metrics = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        # save last checkpoint
        last_path = os.path.join(ckpt_dir, 'last.pth')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'opt_state': opt.state_dict(),
            'val_acc': val_acc,
            'meta': meta,
        }, last_path)

        # save best checkpoint by val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(ckpt_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'opt_state': opt.state_dict(),
                'val_acc': val_acc,
                'meta': meta,
            }, best_path)
            # also save meta JSON for easy inspection
            with open(os.path.join(ckpt_dir, 'run_meta.json'), 'w') as f:
                json.dump(meta, f)
            print(f"Saved new best checkpoint to {best_path} (val_acc={val_acc:.4f})")

    # after training, save metrics and draw plots
    try:
        metrics_df = pd.DataFrame(metrics)
        metrics_csv = os.path.join(ckpt_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_csv, index=False)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], label='train_loss', color='tab:blue')
        ax1.plot(metrics_df['epoch'], metrics_df['val_loss'], label='val_loss', color='tab:orange')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(metrics_df['epoch'], metrics_df['train_acc'], label='train_acc', color='tab:green', linestyle='--')
        ax2.plot(metrics_df['epoch'], metrics_df['val_acc'], label='val_acc', color='tab:red', linestyle='--')
        ax2.set_ylabel('accuracy')
        ax2.legend(loc='upper right')

        plt.title('Training progress')
        plt.tight_layout()
        plot_path = os.path.join(ckpt_dir, 'training.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved training metrics to {metrics_csv} and plot to {plot_path}")
        
        # Generate gradient plots if recording was enabled
        if args.record_gradients and gradient_history:
            try:
                # Gradient norm plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                epochs = range(1, len(gradient_history['avg_gradient_norm']) + 1)
                ax1.plot(epochs, gradient_history['avg_gradient_norm'], 'b-', label='Average Gradient Norm')
                ax1.plot(epochs, gradient_history['max_gradient_norm'], 'r--', label='Max Gradient Norm')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Gradient Norm')
                ax1.set_title('Gradient Norms During Training')
                ax1.legend()
                ax1.grid(True)
                ax1.set_yscale('log')
                
                # Parameter update norm plot
                ax2.plot(epochs, gradient_history['avg_parameter_update_norm'], 'g-', label='Parameter Update Norm')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Update Norm')
                ax2.set_title('Parameter Update Magnitudes')
                ax2.legend()
                ax2.grid(True)
                ax2.set_yscale('log')
                
                plt.tight_layout()
                gradient_plot_path = os.path.join(ckpt_dir, 'gradient_analysis.png')
                fig.savefig(gradient_plot_path)
                plt.close(fig)
                
                # Save gradient history as JSON
                gradient_json_path = os.path.join(ckpt_dir, 'gradient_history.json')
                # Convert numpy arrays to lists for JSON serialization
                gradient_data = {k: [float(x) for x in v] for k, v in gradient_history.items()}
                with open(gradient_json_path, 'w') as f:
                    json.dump(gradient_data, f, indent=2)
                
                print(f"Saved gradient analysis to {gradient_plot_path} and {gradient_json_path}")
            except Exception as e:
                print(f"Failed to save gradient plots: {e}")
                
    except Exception as e:
        print(f"Failed to save/plot metrics: {e}")


if __name__ == "__main__":
    main()
