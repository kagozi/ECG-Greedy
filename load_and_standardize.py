# ============================================================================
# STEP 1: Load Raw ECG Signals and Apply Z-Score Standardization
# ============================================================================
# Run this first to create standardized signal files
# Outputs: train_std.npy, val_std.npy, test_std.npy, labels, scaler

import os
import pickle
import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from tqdm import tqdm
import wfdb

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = '../datasets/ECG/'
PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
SAMPLING_RATE = 100

os.makedirs(PROCESSED_PATH, exist_ok=True)

print("="*80)
print("STEP 1: LOAD AND STANDARDIZE ECG SIGNALS")
print("="*80)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ptbxl_dataset(data_path, processed_path, sampling_rate=100):
    """Load PTB-XL dataset"""
    print("\nLoading metadata...")
    Y = pd.read_csv(data_path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    print("Loading raw signals...")
    X = load_raw_signals(Y, sampling_rate, data_path, processed_path)
    
    return X, Y


def load_raw_signals(df, sampling_rate, data_path, processed_path):
    """Load raw ECG signals with caching"""
    cache_file = os.path.join(processed_path, f'raw{sampling_rate}.npy')
    
    if os.path.exists(cache_file):
        print(f"Loading cached raw signals from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
    else:
        print(f"Loading and caching raw signals at {sampling_rate}Hz...")
        if sampling_rate == 100:
            data = [wfdb.rdsamp(data_path + f) for f in tqdm(df.filename_lr)]
        else:
            data = [wfdb.rdsamp(data_path + f) for f in tqdm(df.filename_hr)]
        
        data = np.array([signal for signal, meta in data])
        print(f"Saving to cache: {cache_file}")
        np.save(cache_file, data)
    
    print(f"Loaded signals shape: {data.shape}")
    return data


def aggregate_diagnostic_labels(df, scp_statements_path):
    """Aggregate SCP codes into superclasses"""
    print("\nAggregating diagnostic labels...")
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
    print("\nPreparing multi-label encoding...")
    mlb = MultiLabelBinarizer()
    
    counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
    counts = counts[counts > min_samples]
    
    Y.diagnostic_superclass = Y.diagnostic_superclass.apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )
    Y['diagnostic_len'] = Y.diagnostic_superclass.apply(lambda x: len(x))
    
    # Filter samples with at least one label
    X = X[Y.diagnostic_len > 0]
    Y = Y[Y.diagnostic_len > 0]
    
    mlb.fit(Y.diagnostic_superclass.values)
    y = mlb.transform(Y.diagnostic_superclass.values)
    
    print(f"Classes: {mlb.classes_}")
    print(f"Number of samples: {len(X)}")
    print(f"Label shape: {y.shape}")
    
    return X, Y, y, mlb

# ============================================================================
# STANDARDIZATION FUNCTIONS
# ============================================================================

def standardize_signals(X_train, X_val, X_test):
    """
    Apply Z-score normalization to ECG signals
    Fit on training data, apply to all sets
    """
    print("\nFitting StandardScaler on training data...")
    
    # Fit scaler on flattened training data
    ss = StandardScaler()
    train_flat = np.vstack(X_train).flatten()[:, np.newaxis].astype(float)
    ss.fit(train_flat)
    
    print(f"Scaler stats - Mean: {ss.mean_[0]:.4f}, Std: {ss.scale_[0]:.4f}")
    
    # Apply to all sets
    print("Standardizing training set...")
    X_train_std = apply_standardizer(X_train, ss)
    
    print("Standardizing validation set...")
    X_val_std = apply_standardizer(X_val, ss)
    
    print("Standardizing test set...")
    X_test_std = apply_standardizer(X_test, ss)
    
    return X_train_std, X_val_std, X_test_std, ss


def apply_standardizer(X, ss):
    """Apply standardization to signals"""
    X_standardized = []
    for x in tqdm(X):
        x_shape = x.shape
        x_std = ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape)
        X_standardized.append(x_std)
    return np.array(X_standardized)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Load dataset
    print("\n[1/5] Loading PTB-XL dataset...")
    X, Y = load_ptbxl_dataset(DATA_PATH, PROCESSED_PATH, SAMPLING_RATE)
    
    # Process labels
    print("\n[2/5] Processing labels...")
    Y = aggregate_diagnostic_labels(Y, DATA_PATH + 'scp_statements.csv')
    X, Y, y, mlb = prepare_labels(X, Y, min_samples=0)
    
    # Split data using stratified folds
    print("\n[3/5] Splitting data by stratified folds...")
    train_mask = Y.strat_fold <= 8
    val_mask = Y.strat_fold == 9
    test_mask = Y.strat_fold == 10
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    Y_train = Y[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    Y_val = Y[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    Y_test = Y[test_mask]
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Standardize signals
    print("\n[4/5] Applying Z-score standardization...")
    X_train_std, X_val_std, X_test_std, scaler = standardize_signals(
        X_train, X_val, X_test
    )
    
    # Save standardized signals and metadata
    print("\n[5/5] Saving standardized data...")
    
    # Save signals (memory-mapped for efficiency)
    np.save(os.path.join(PROCESSED_PATH, 'train_standardized.npy'), X_train_std)
    np.save(os.path.join(PROCESSED_PATH, 'val_standardized.npy'), X_val_std)
    np.save(os.path.join(PROCESSED_PATH, 'test_standardized.npy'), X_test_std)
    
    print(f"✓ Saved train_standardized.npy: {X_train_std.shape}")
    print(f"✓ Saved val_standardized.npy: {X_val_std.shape}")
    print(f"✓ Saved test_standardized.npy: {X_test_std.shape}")
    
    # Save labels
    np.save(os.path.join(PROCESSED_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_PATH, 'y_test.npy'), y_test)
    
    print(f"✓ Saved label arrays")
    
    # Save scaler and label encoder
    with open(os.path.join(PROCESSED_PATH, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(PROCESSED_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(mlb, f)
    
    print(f"✓ Saved scaler and label encoder")
    
    # Save metadata
    metadata = {
        'num_classes': len(mlb.classes_),
        'classes': mlb.classes_.tolist(),
        'train_size': len(X_train_std),
        'val_size': len(X_val_std),
        'test_size': len(X_test_std),
        'sampling_rate': SAMPLING_RATE,
        'signal_shape': X_train_std[0].shape,
        'scaler_mean': float(scaler.mean_[0]),
        'scaler_std': float(scaler.scale_[0])
    }
    
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Saved metadata")
    
    print("\n" + "="*80)
    print("STEP 1 COMPLETE!")
    print("="*80)
    print(f"\nAll files saved to: {PROCESSED_PATH}")
    print("\nNext step: Run 2_generate_cwt.py to create scalograms and phasograms")


if __name__ == '__main__':
    main()