# benchmark.py
# ============================================================================
# XRESNET1D BENCHMARK REPLICATION - PURE PYTORCH
# ============================================================================

import os
import pickle
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import fbeta_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wfdb

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_ptbxl_dataset(data_path, processed_path, sampling_rate=100):
    """Load PTB-XL dataset"""
    # Load annotations
    Y = pd.read_csv(data_path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load raw signals
    X = load_raw_signals(Y, sampling_rate, data_path, processed_path)
    
    return X, Y


def load_raw_signals(df, sampling_rate, data_path, processed_path):
    """Load raw ECG signals"""
    os.makedirs(processed_path, exist_ok=True)
    cache_file = os.path.join(processed_path, f'raw{sampling_rate}.npy')
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Loading and caching raw signals at {sampling_rate}Hz")
        if sampling_rate == 100:
            data = [wfdb.rdsamp(data_path + f) for f in tqdm(df.filename_lr)]
        else:  # 500 Hz
            data = [wfdb.rdsamp(data_path + f) for f in tqdm(df.filename_hr)]
        
        data = np.array([signal for signal, meta in data])
        np.save(cache_file, data)
    
    return data

# ============================================================================
# STEP 2: LABEL PROCESSING
# ============================================================================

def aggregate_diagnostic_labels(df, scp_statements_path):
    """Aggregate SCP codes into superclasses"""
    aggregation_df = pd.read_csv(scp_statements_path, index_col=0)
    diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
    
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in diag_agg_df.index:
                c = diag_agg_df.loc[key].diagnostic_class
                if str(c) != 'nan':
                    tmp.append(c)
        return list(set(tmp))
    
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    df['diagnostic_len'] = df.diagnostic_superclass.apply(lambda x: len(x))
    
    return df

def prepare_labels(X, Y, min_samples=0):
    """Convert to multi-hot encoding"""
    mlb = MultiLabelBinarizer()
    
    # Filter by minimum samples
    counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
    counts = counts[counts > min_samples]
    
    Y.diagnostic_superclass = Y.diagnostic_superclass.apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )
    Y['diagnostic_len'] = Y.diagnostic_superclass.apply(lambda x: len(x))
    
    # Remove samples with no labels
    X = X[Y.diagnostic_len > 0]
    Y = Y[Y.diagnostic_len > 0]
    
    # Transform to multi-hot
    mlb.fit(Y.diagnostic_superclass.values)
    y = mlb.transform(Y.diagnostic_superclass.values)
    
    print(f"Classes: {mlb.classes_}")
    print(f"Number of samples: {len(X)}")
    print(f"Class distribution:\n{pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()}")
    
    return X, Y, y, mlb

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

def preprocess_signals(X_train, X_val, X_test):
    """Standardize signals"""
    # Fit on training data
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    
    # Apply to all sets
    X_train_scaled = apply_standardizer(X_train, ss)
    X_val_scaled = apply_standardizer(X_val, ss)
    X_test_scaled = apply_standardizer(X_test, ss)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, ss

def apply_standardizer(X, ss):
    """Apply standardization to signals"""
    X_tmp = []
    for x in tqdm(X, desc="Standardizing"):
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    return np.array(X_tmp)

# ============================================================================
# STEP 4: XRESNET1D ARCHITECTURE (PURE PYTORCH)
# ============================================================================

class BasicBlock1d(nn.Module):
    """Basic ResNet block for 1D signals"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class XResNet1d101(nn.Module):
    """XResNet1d-101 architecture"""
    
    def __init__(self, input_channels=12, num_classes=5, base_filters=64):
        super().__init__()
        
        self.in_channels = base_filters
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=7, stride=2, 
                     padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet blocks [3, 4, 23, 3] for ResNet-101
        self.layer1 = self._make_layer(base_filters, 3, stride=1)
        self.layer2 = self._make_layer(base_filters*2, 4, stride=2)
        self.layer3 = self._make_layer(base_filters*4, 23, stride=2)
        self.layer4 = self._make_layer(base_filters*8, 3, stride=2)
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(base_filters*8*2, num_classes)  # *2 for concat pooling
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        
        # First block may have stride > 1
        layers.append(BasicBlock1d(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1d(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Concatenated pooling
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        
        x = x.flatten(1)
        x = self.fc(x)
        
        return x

# ============================================================================
# STEP 5: PYTORCH DATASET
# ============================================================================

class ECGDataset(Dataset):
    """PyTorch Dataset for ECG signals"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 2, 1)  # (N, time, channels) -> (N, channels, time)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# STEP 6: TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return running_loss / len(dataloader.dataset), all_preds, all_labels

# ============================================================================
# STEP 7: EVALUATION METRICS
# ============================================================================

def compute_metrics(y_true, y_pred, y_scores):
    """Compute evaluation metrics"""
    # Macro AUC
    macro_auc = roc_auc_score(y_true, y_scores, average='macro')
    
    # F-beta score (beta=2 as in benchmark)
    f_beta = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
    
    return {
        'macro_auc': macro_auc,
        'f_beta_macro': f_beta
    }

def find_optimal_thresholds(y_true, y_scores):
    """Find optimal thresholds per class"""
    from sklearn.metrics import roc_curve
    
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, threshold = roc_curve(y_true[:, i], y_scores[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        thresholds.append(threshold[optimal_idx])
    
    return np.array(thresholds)

def apply_thresholds(y_scores, thresholds):
    """Apply class-wise thresholds"""
    y_pred = (y_scores > thresholds).astype(int)
    
    # If no prediction, take the maximum
    for i, pred in enumerate(y_pred):
        if pred.sum() == 0:
            y_pred[i, np.argmax(y_scores[i])] = 1
    
    return y_pred

def plot_confusion_matrices(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix for each class"""
    n_classes = y_true.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
    
    if n_classes == 1:
        axes = [axes]
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{class_name}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_confusion_matrix_all_classes(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix - All Classes"):
    """
    Plots a single confusion matrix showing all 5 classes together.
    For multi-label classification, we convert to multi-class by taking the class with highest probability.
    """
    # Convert multi-label to multi-class by taking the class with highest probability
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# STEP 8: MAIN TRAINING PIPELINE
# ============================================================================

def main():
    # Configuration
    DATA_PATH = '/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'  # UPDATE THIS
    PROCESSED_PATH = '/kaggle/working/'
    SAMPLING_RATE = 100
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("XRESNET1D101 BENCHMARK REPLICATION")
    print("="*80)
    
    # Step 1: Load data
    print("\n[1/8] Loading PTB-XL dataset...")
    X, Y = load_ptbxl_dataset(DATA_PATH, PROCESSED_PATH, sampling_rate=100)
    
    # Step 2: Process labels
    print("\n[2/8] Processing labels...")
    Y = aggregate_diagnostic_labels(Y, DATA_PATH + 'scp_statements.csv')
    X, Y, y, mlb = prepare_labels(X, Y, min_samples=0)
    
    # Step 3: Split data (folds 1-8: train, 9: val, 10: test)
    print("\n[3/8] Splitting data...")
    X_train = X[Y.strat_fold <= 8]
    y_train = y[Y.strat_fold <= 8]
    
    X_val = X[Y.strat_fold == 9]
    y_val = y[Y.strat_fold == 9]
    
    X_test = X[Y.strat_fold == 10]
    y_test = y[Y.strat_fold == 10]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 4: Preprocess
    print("\n[4/8] Preprocessing signals...")
    X_train, X_val, X_test, scaler = preprocess_signals(X_train, X_val, X_test)
    
    # Step 5: Create datasets
    print("\n[5/8] Creating PyTorch datasets...")
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Step 6: Create model
    print("\n[6/8] Creating XResNet1d101 model...")
    model = XResNet1d101(input_channels=12, num_classes=len(mlb.classes_)).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 7: Train
    print("\n[7/8] Training model...")
    best_val_auc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
        
        # Compute metrics
        thresholds = find_optimal_thresholds(val_labels, val_preds)
        val_pred_binary = apply_thresholds(val_preds, thresholds)
        val_metrics = compute_metrics(val_labels, val_pred_binary, val_preds)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_metrics['macro_auc']:.4f} | Val F-beta: {val_metrics['f_beta_macro']:.4f}")
        
        # Save best model
        if val_metrics['macro_auc'] > best_val_auc:
            best_val_auc = val_metrics['macro_auc']
            torch.save(model.state_dict(), 'best_xresnet1d101.pth')
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        
        scheduler.step(val_loss)
    
    # Step 8: Test
    print("\n[8/8] Testing on test set...")
    model.load_state_dict(torch.load('best_xresnet1d101.pth'))
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
    
    # Optimize thresholds on validation set
    _, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)
    thresholds = find_optimal_thresholds(val_labels, val_preds)
    
    # Apply to test set
    test_pred_binary = apply_thresholds(test_preds, thresholds)
    test_metrics = compute_metrics(test_labels, test_pred_binary, test_preds)
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Test AUC (Macro): {test_metrics['macro_auc']:.4f}")
    print(f"Test F-beta (Macro): {test_metrics['f_beta_macro']:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrices(test_labels, test_pred_binary, mlb.classes_, 
                           save_path='confusion_matrices.png')
    
    print("\n✓ Training complete!")

if __name__ == '__main__':
    main()