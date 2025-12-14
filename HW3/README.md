# Shallow MLP trainer (PyTorch)

1) Shallow training — Adult (2 hidden layers, weighted loss only):

```bash
python3 train.py \
	--dataset adult \
	--hidden-sizes 128 64 \
	--epochs 20 \
	--batch-size 128 \
	--lr 1e-3 \
	--dropout 0.1 \
	--activation relu \
	--loss cross_entropy \
	--val-size 0.1 \
	--checkpoint-dir checkpoints/mlp_adult_shallow_weighted
```

2) Shallow training — Adult (use weighted sampler to balance batches):

```bash
python3 train.py \
	--dataset adult \
	--hidden-sizes 256 128 \
	--epochs 20 \
	--batch-size 128 \
	--lr 1e-3 \
	--dropout 0.1 \
	--activation relu \
	--loss cross_entropy \
	--val-size 0.1 \
	--use-weighted-sampler \
	--checkpoint-dir checkpoints/mlp_adult_shallow_sampler \
  --record-gradients
```

3) Deep training — Adult (5 hidden layers):

```bash
python3 train.py \
	--dataset adult \
	--hidden-sizes 256 128 64 32 16 \
	--epochs 20 \
	--batch-size 128 \
	--lr 5e-4 \
	--dropout 0.3 \
	--activation relu \
	--loss cross_entropy \
	--val-size 0.1 \
	--use-weighted-sampler \
	--checkpoint-dir checkpoints/mlp_adult_deep \
	--record-gradients
```

4) Shallow training — MNIST (2 hidden layers, ReLU):

```bash
python3 train.py \
	--dataset mnist \
	--hidden-sizes 512 256 \
	--epochs 10 \
	--batch-size 128 \
	--lr 1e-3 \
	--dropout 0.2 \
	--activation relu \
	--loss cross_entropy \
	--val-size 0.1 \
	--checkpoint-dir checkpoints/mlp_mnist_shallow_relu \
  --record-gradients
```

4b) Shallow training — MNIST (2 hidden layers, Sigmoid):

```bash
python3 train.py \
	--dataset mnist \
	--hidden-sizes 512 256 \
	--epochs 10 \
	--batch-size 128 \
	--lr 1e-3 \
	--dropout 0.2 \
	--activation sigmoid \
	--loss cross_entropy \
	--val-size 0.1 \
	--checkpoint-dir checkpoints/mlp_mnist_shallow_sigmoid \
  --record-gradients
```

5) Deep training — MNIST (5 hidden layers, ReLU):

```bash
python3 train.py \
	--dataset mnist \
	--hidden-sizes 1024 512 256 128 64 \
	--epochs 20 \
	--batch-size 128 \
	--lr 5e-4 \
	--dropout 0.5 \
	--activation relu \
	--loss cross_entropy \
	--val-size 0.1 \
	--checkpoint-dir checkpoints/mlp_mnist_deep_relu \
	--record-gradients
```

5b) Deep training — MNIST (5 hidden layers, Sigmoid):

```bash
python3 train.py \
	--dataset mnist \
	--hidden-sizes 1024 512 256 128 64 \
	--epochs 20 \
	--batch-size 128 \
	--lr 5e-4 \
	--dropout 0.5 \
	--activation sigmoid \
	--loss cross_entropy \
	--val-size 0.1 \
	--checkpoint-dir checkpoints/mlp_mnist_deep_sigmoid \
	--record-gradients
```

6) Testing — Adult models (evaluate specific checkpoints):

```bash
# Test shallow weighted model
python3 test.py \
	--ckpt checkpoints/mlp_adult_shallow_weighted/best.pth \
	--dataset adult \
	--batch-size 256 \
	--out checkpoints/mlp_adult_shallow_weighted/predictions.csv \
	--plot-dir checkpoints/mlp_adult_shallow_weighted

# Test shallow sampler model
python3 test.py \
	--ckpt checkpoints/mlp_adult_shallow_sampler/best.pth \
	--dataset adult \
	--batch-size 256 \
	--out checkpoints/mlp_adult_shallow_sampler/predictions.csv \
	--plot-dir checkpoints/mlp_adult_shallow_sampler

# Test deep model
python3 test.py \
	--ckpt checkpoints/mlp_adult_deep/best.pth \
	--dataset adult \
	--batch-size 256 \
	--out checkpoints/mlp_adult_deep/predictions.csv \
	--plot-dir checkpoints/mlp_adult_deep
```

7) Testing — MNIST models (evaluate specific checkpoints):

```bash
# Test shallow ReLU model
python3 test.py \
	--ckpt checkpoints/mlp_mnist_shallow_relu/best.pth \
	--dataset mnist \
	--batch-size 256 \
	--out checkpoints/mlp_mnist_shallow_relu/predictions.csv \
	--plot-dir checkpoints/mlp_mnist_shallow_relu

# Test shallow Sigmoid model
python3 test.py \
	--ckpt checkpoints/mlp_mnist_shallow_sigmoid/best.pth \
	--dataset mnist \
	--batch-size 256 \
	--out checkpoints/mlp_mnist_shallow_sigmoid/predictions.csv \
	--plot-dir checkpoints/mlp_mnist_shallow_sigmoid

# Test deep ReLU model
python3 test.py \
	--ckpt checkpoints/mlp_mnist_deep_relu/best.pth \
	--dataset mnist \
	--batch-size 256 \
	--out checkpoints/mlp_mnist_deep_relu/predictions.csv \
	--plot-dir checkpoints/mlp_mnist_deep_relu

# Test deep Sigmoid model
python3 test.py \
	--ckpt checkpoints/mlp_mnist_deep_sigmoid/best.pth \
	--dataset mnist \
	--batch-size 256 \
	--out checkpoints/mlp_mnist_deep_sigmoid/predictions.csv \
	--plot-dir checkpoints/mlp_mnist_deep_sigmoid
```

## XGBoost Baseline Training

Train an XGBoost classifier as a baseline comparison:

### Adult Dataset (Binary Classification)

1) training

```bash
python3 train_xgboost.py \
  --dataset adult \
  --n-estimators 100 \
  --max-depth 3 \
  --learning-rate 0.1 \
  --batch-size 256 \
  --plot-dir ./checkpoints/xgb_adult_shallow \
  --seed 42
```

```bash
python3 train_xgboost.py \
  --dataset adult \
  --n-estimators 500 \
  --max-depth 10 \
  --learning-rate 0.05 \
  --batch-size 256 \
  --plot-dir ./checkpoints/xgb_adult_deep \
  --seed 42
```

### MNIST Dataset (Multi-class Classification)

1) training

```bash
python3 train_xgboost.py \
  --dataset mnist \
  --n-estimators 200 \
  --max-depth 6 \
  --learning-rate 0.1 \
  --batch-size 512 \
  --plot-dir ./checkpoints/xgb_mnist_shallow \
  --seed 42
```

```bash
python3 train_xgboost.py \
  --dataset mnist \
  --n-estimators 800 \
  --max-depth 12 \
  --learning-rate 0.05 \
  --batch-size 512 \
  --plot-dir ./checkpoints/xgb_mnist_deep \
  --seed 42
```

2) testing

```bash
# Test Adult XGBoost models
python3 test.py \
	--ckpt ./checkpoints/best.pth \
	--xgb ./checkpoints/xgb_adult_shallow/xgb_model.pkl \
	--dataset adult \
	--out ./checkpoints/xgb_adult_shallow/xgb_preds.csv \
	--plot-dir ./checkpoints/xgb_adult_shallow
	--threshold 0.7
```

```bash
python3 test.py \
	--ckpt ./checkpoints/best.pth \
	--xgb ./checkpoints/xgb_adult_deep/xgb_model.pkl \
	--dataset adult \
	--out ./checkpoints/xgb_adult_deep/xgb_preds.csv \
	--plot-dir ./checkpoints/xgb_adult_deep \
	--threshold 0.7
```

```bash
# Test MNIST XGBoost models
python3 test.py \
	--ckpt ./checkpoints/best.pth \
	--xgb ./checkpoints/xgb_mnist_shallow/xgb_model.pkl \
	--dataset mnist \
	--out ./checkpoints/xgb_mnist_shallow/xgb_preds.csv \
	--plot-dir ./checkpoints/xgb_mnist_shallow
```

```bash
python3 test.py \
	--ckpt ./checkpoints/best.pth \
	--xgb ./checkpoints/xgb_mnist_deep/xgb_model.pkl \
	--dataset mnist \
	--out ./checkpoints/xgb_mnist_deep/xgb_preds.csv \
	--plot-dir ./checkpoints/xgb_mnist_deep
```

## CNN Training

Train CNN models on MNIST

1) training

```bash
python3 train_cnn.py \
  --dataset mnist \
  --num-conv-layers 2 \
  --base-channels 16 \
  --channel-multiplier 2.0 \
  --fc-sizes 128 64 \
  --epochs 10 \
  --batch-size 128 \
  --lr 0.001 \
  --dropout 0.3 \
  --checkpoint-dir checkpoints/cnn_mnist_shallow \
  --plot \
  --record-gradients
```

```bash
python3 train_cnn.py \
  --dataset mnist \
  --num-conv-layers 5 \
  --base-channels 64 \
  --channel-multiplier 2.0 \
  --fc-sizes 512 256 128 \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.0005 \
  --dropout 0.5 \
  --optimizer adamw \
  --scheduler cosine \
  --checkpoint-dir checkpoints/cnn_mnist_deep \
  --plot \
  --record-gradients
```

2) testing

```bash
# Test shallow CNN
python3 test.py \
  --ckpt checkpoints/cnn_mnist_shallow/best_cnn.pth \
  --dataset mnist \
  --batch-size 256 \
  --out checkpoints/cnn_mnist_shallow/predictions.csv \
  --plot-dir checkpoints/cnn_mnist_shallow

# Test deep CNN
python3 test.py \
  --ckpt checkpoints/cnn_mnist_deep/best_cnn.pth \
  --dataset mnist \
  --batch-size 256 \
  --out checkpoints/cnn_mnist_deep/predictions.csv \
  --plot-dir checkpoints/cnn_mnist_deep
```

Train CNN models on Adult

1) Training 
```bash
python3 train_cnn.py \
  --dataset adult \
  --num-conv-layers 2 \
  --base-channels 16 \
  --channel-multiplier 1.5 \
  --fc-sizes 64 32 \
  --epochs 12 \
  --batch-size 256 \
  --lr 0.001 \
  --dropout 0.4 \
  --class-weight \
  --one-hot-only \
  --checkpoint-dir checkpoints/cnn_adult_shallow \
  --plot \
  --record-gradients
```

```bash
# Train deep CNN on Adult dataset (4 conv layers, feature selection)
python3 train_cnn.py \
  --dataset adult \
  --num-conv-layers 4 \
  --base-channels 32 \
  --channel-multiplier 2.0 \
  --fc-sizes 256 128 64 \
  --epochs 20 \
  --batch-size 128 \
  --lr 0.0008 \
  --dropout 0.5 \
  --class-weight \
  --feature-selection pearson \
  --feature-threshold 0.02 \
  --optimizer adamw \
  --scheduler step \
  --checkpoint-dir checkpoints/cnn_adult_deep \
  --plot \
  --record-gradients
```

2) Testing

```bash
# Test shallow CNN
python3 test.py \
  --ckpt checkpoints/cnn_adult_shallow/best_cnn.pth \
  --dataset adult \
  --batch-size 256 \
  --out checkpoints/cnn_adult_shallow/predictions.csv \
  --plot-dir checkpoints/cnn_adult_shallow

# Test deep CNN  
python3 test.py \
  --ckpt checkpoints/cnn_adult_deep/best_cnn.pth \
  --dataset adult \
  --batch-size 256 \
  --out checkpoints/cnn_adult_deep/predictions.csv \
  --plot-dir checkpoints/cnn_adult_deep
```

## SVM Training

Train Support Vector Machine models. SVMs excel at binary classification (Adult) and handle multi-class (MNIST) with one-vs-rest or one-vs-one strategies.

### Adult Dataset (Binary Classification - SVM's Strength)

1) Linear SVM (fast, interpretable):

```bash
python3 train_svm.py \
  --dataset adult \
  --kernel linear \
  --C 1.0 \
  --class-weight balanced \
  --one-hot-only \
  --checkpoint-dir checkpoints/svm_adult_linear
```

2) RBF SVM with grid search (best performance):

```bash
python3 train_svm.py \
  --dataset adult \
  --kernel rbf \
  --grid-search \
  --cv-folds 5 \
  --class-weight balanced \
  --feature-selection pearson \
  --feature-threshold 0.02 \
  --checkpoint-dir checkpoints/svm_adult_rbf_tuned
```

3) Quick RBF SVM (no grid search):

```bash
python3 train_svm.py \
  --dataset adult \
  --kernel rbf \
  --C 1.0 \
  --gamma scale \
  --class-weight balanced \
  --one-hot-only \
  --checkpoint-dir checkpoints/svm_adult_rbf
```

### MNIST Dataset (Multi-class - More Challenging for SVM)

1) Linear SVM (fast baseline):

```bash
python3 train_svm.py \
  --dataset mnist \
  --kernel linear \
  --C 1.0 \
  --multiclass ovr \
  --max-samples 10000 \
  --checkpoint-dir checkpoints/svm_mnist_linear
```

2) RBF SVM with grid search (best but slow):

```bash
python3 train_svm.py \
  --dataset mnist \
  --kernel rbf \
  --grid-search \
  --cv-folds 3 \
  --multiclass ovr \
  --max-samples 5000 \
  --checkpoint-dir checkpoints/svm_mnist_rbf_tuned
```

3) Polynomial SVM (alternative non-linear):

```bash
python3 train_svm.py \
  --dataset mnist \
  --kernel poly \
  --degree 3 \
  --C 1.0 \
  --multiclass ovo \
  --max-samples 8000 \
  --checkpoint-dir checkpoints/svm_mnist_poly
```

### Testing SVM Models

```bash
# Test Adult SVM models
python3 test.py \
  --xgb checkpoints/svm_adult_linear/svm_model.pkl \
  --dataset adult \
  --batch-size 256 \
  --out checkpoints/svm_adult_linear/predictions.csv \
  --plot-dir checkpoints/svm_adult_linear

python3 test.py \
  --xgb checkpoints/svm_adult_rbf/svm_model.pkl \
  --dataset adult \
  --batch-size 256 \
  --out checkpoints/svm_adult_rbf/predictions.csv \
  --plot-dir checkpoints/svm_adult_rbf

python3 test.py \
  --xgb checkpoints/svm_adult_rbf_tuned/svm_model.pkl \
  --dataset adult \
  --batch-size 256 \
  --out checkpoints/svm_adult_rbf_tuned/predictions.csv \
  --plot-dir checkpoints/svm_adult_rbf_tuned

# Test MNIST SVM models  
python3 test.py \
  --xgb checkpoints/svm_mnist_linear/svm_model.pkl \
  --dataset mnist \
  --batch-size 256 \
  --out checkpoints/svm_mnist_linear/predictions.csv \
  --plot-dir checkpoints/svm_mnist_linear

python3 test.py \
  --xgb checkpoints/svm_mnist_poly/svm_model.pkl \
  --dataset mnist \
  --batch-size 256 \
  --out checkpoints/svm_mnist_poly/predictions.csv \
  --plot-dir checkpoints/svm_mnist_poly

python3 test.py \
  --xgb checkpoints/svm_mnist_rbf_tuned/svm_model.pkl \
  --dataset mnist \
  --batch-size 256 \
  --out checkpoints/svm_mnist_rbf_tuned/predictions.csv \
  --plot-dir checkpoints/svm_mnist_rbf_tuned
```