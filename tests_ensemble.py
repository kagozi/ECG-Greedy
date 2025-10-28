import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, 
    fbeta_score, classification_report
)
from tqdm import tqdm

# Import your models
from models import (CWT2DCNN, DualStreamCNN, ViTFusionECG, 
                    SwinTransformerECG, SwinTransformerEarlyFusion, 
                    ViTLateFusion, EfficientNetLateFusion, 
                    SwinTransformerLateFusion, HybridSwinTransformerECG ,HybridSwinTransformerEarlyFusion, 
                    HybridSwinTransformerLateFusion,
                    EfficientNetEarlyFusion, EfficientNetLateFusion,
                    EfficientNetFusionECG, ResNet50EarlyFusion, 
                    ResNet50LateFusion,
                    ResNet50ECG
)
from benchmark import XResNet1d101, load_ptbxl_dataset, aggregate_diagnostic_labels, preprocess_signals, prepare_labels

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/results/'
BASELINE_RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/results/'
DATA_PATH = '../datasets/ECG/'
ENSEMBLE_PATH = os.path.join(RESULTS_PATH, 'ensemble_results/')
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(ENSEMBLE_PATH, exist_ok=True)

print("="*80)
print("CONFUSION MATRICES & ENSEMBLE ANALYSIS (with ResNet1D Baseline)")
print("="*80)
print(f"Device: {DEVICE}")



# ============================================================================
# DATASET CLASSES
# ============================================================================

class CWTDataset(Dataset):
    """Memory-efficient dataset for CWT data"""
    
    def __init__(self, scalo_path, phaso_path, labels, mode='scalogram'):
        self.scalograms = np.load(scalo_path, mmap_mode='r')
        self.phasograms = np.load(phaso_path, mmap_mode='r')
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
        phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
        label = self.labels[idx]
        
        if self.mode == 'scalogram':
            return scalo, label
        elif self.mode == 'phasogram':
            return phaso, label
        elif self.mode == 'both':
            return (scalo, phaso), label
        elif self.mode == 'fusion':
            fused = torch.cat([scalo, phaso], dim=0)
            return fused, label

class ECGDataset(Dataset):
    """Dataset for raw 1D ECG signals (for ResNet1D baseline)"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 2, 1)  # (N, time, channels) -> (N, channels, time)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

def load_model_from_config(config, num_classes):
    """Load model architecture based on config"""
    mode = config['mode']
    # model_type = config['model']
    raw_model_type = config['model']
    # normalize: take prefix before any '-' or '_'
    model_type = raw_model_type.split('-')[0].split('_')[0]
    adapter_strategy = config.get('adapter', 'learned')
        
    if config['model'] == 'DualStream':
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif config['model'] == 'CWT2DCNN':
        # Adjust channels for fusion mode (24 channels = 12 scalo + 12 phaso)
        num_ch = 24 if mode == 'fusion' else 12
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif config['model'] == 'ViTFusionECG':
        model = ViTFusionECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'SwinTransformerECG':
        model = SwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'SwinTransformerEarlyFusion':
        model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'SwinTransformerLateFusion':
        model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'ViTLateFusion':
        model = ViTLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'HybridSwinTransformerECG':
        model = HybridSwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'HybridSwinTransformerEarlyFusion':
        model = HybridSwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'HybridSwinTransformerLateFusion':
        model = HybridSwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
     # EfficientNet variants
    elif config['model'] == 'EfficientNetFusionECG':
        model = EfficientNetFusionECG(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'EfficientNetEarlyFusion':
        model = EfficientNetEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'EfficientNetLateFusion':
        model = EfficientNetLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    # ResNet50 variants
    elif config['model'] == 'ResNet50EarlyFusion':
        model = ResNet50EarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'ResNet50LateFusion':
        model = ResNet50LateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'ResNet50ECG':
        model = ResNet50ECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'ResNet50EarlyFusion':
        model = ResNet50EarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'ResNet50LateFusion':
        model = ResNet50LateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif model_type == 'XResNet1d101':
        # ResNet1D baseline
        model = XResNet1d101(
            input_channels=12, 
            num_classes=num_classes,
            stem_ks=config.get('stem_ks', 7),
            block_ks=config.get('block_ks', 3),
            drop_rate=config.get('drop_rate', 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(DEVICE)


def load_model_checkpoint_safely(model, checkpoint_path, device):
    """
    Safely load model checkpoint with compatibility handling for structure mismatches
    
    Args:
        model: PyTorch model instance
        checkpoint_path: Path to checkpoint file
        device: torch.device
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Try direct loading first
        try:
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}

            for k, v in state_dict.items():
                if k.startswith("adapter.weight"):
                    # rename old key → expected new key
                    new_state_dict["adapter.adapter.weight"] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            model.load_state_dict(state_dict, strict=True)
            return True
        except RuntimeError as e:
            error_msg = str(e)
            
            # Handle adapter key mismatch (adapter.weight vs adapter.adapter.weight)
            if "adapter" in error_msg and ("Missing key" in error_msg or "Unexpected key" in error_msg):
                print(f"  ⚠️  Adapter key mismatch detected, attempting to fix...")
                
                # Create a new state dict with corrected keys
                new_state_dict = {}
                model_keys = set(model.state_dict().keys())
                
                for key, value in state_dict.items():
                    # Try to map old adapter keys to new structure
                    if "adapter.weight" in key and "adapter.adapter.weight" not in key:
                        # Old structure: adapter.weight -> New structure: adapter.adapter.weight
                        new_key = key.replace("adapter.weight", "adapter.adapter.weight")
                        if new_key in model_keys:
                            new_state_dict[new_key] = value
                            print(f"      Remapped: {key} -> {new_key}")
                            continue
                    
                    # Try reverse mapping (if model expects old structure but checkpoint has new)
                    if "adapter.adapter.weight" in key:
                        old_key = key.replace("adapter.adapter.weight", "adapter.weight")
                        if old_key in model_keys:
                            new_state_dict[old_key] = value
                            print(f"      Remapped: {key} -> {old_key}")
                            continue
                    
                    # Keep key as-is if it matches
                    if key in model_keys:
                        new_state_dict[key] = value
                
                # Try loading with remapped keys
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                
                if missing_keys:
                    print(f"  ⚠️  Missing keys after remapping: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                if unexpected_keys:
                    print(f"  ⚠️  Unexpected keys after remapping: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                
                # Check if critical keys are missing
                critical_missing = [k for k in missing_keys if not k.startswith('adapter')]
                if critical_missing:
                    print(f"  ❌ Critical keys missing: {critical_missing[:3]}")
                    return False
                
                print(f"  ✓ Loaded with remapped adapter keys")
                return True
            
            # Handle other mismatches with non-strict loading
            elif "Missing key" in error_msg or "Unexpected key" in error_msg:
                print(f"  ⚠️  State dict mismatch, attempting non-strict loading...")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                # Check if too many keys are missing
                if len(missing_keys) > len(model.state_dict()) * 0.1:  # More than 10% missing
                    print(f"  ❌ Too many missing keys ({len(missing_keys)}), skipping model")
                    return False
                
                if missing_keys:
                    print(f"  ⚠️  Missing {len(missing_keys)} keys: {list(missing_keys)[:3]}...")
                if unexpected_keys:
                    print(f"  ⚠️  Unexpected {len(unexpected_keys)} keys: {list(unexpected_keys)[:3]}...")
                
                print(f"  ✓ Loaded with {len(missing_keys)} missing and {len(unexpected_keys)} unexpected keys")
                return True
            else:
                raise
    
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        return False


def evaluate_all_models(metadata, X_test_raw, y_test):
    """Evaluate all trained models and generate confusion matrices"""
    
    print("\n[1/3] Loading all trained models...")
    
    all_model_results = {}
    y_true = y_test
    
    # ========================================================================
    # PART 1: Evaluate ResNet1D Baseline
    # ========================================================================
    
    baseline_checkpoint = os.path.join(BASELINE_RESULTS_PATH, 'best_xresnet1d101.pth')
    
    if os.path.exists(baseline_checkpoint):
        print(f"\n{'='*60}")
        print(f"Evaluating: ResNet1D-Baseline")
        print(f"{'='*60}")
        
        # Create dataset for raw signals
        test_dataset_raw = ECGDataset(X_test_raw, y_test)
        test_loader_raw = DataLoader(
            test_dataset_raw, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # Load model
        baseline_config = {
            'model': 'XResNet1d101',
            'mode': 'raw',
            'stem_ks': 7,
            'block_ks': 3,
            'drop_rate': 0.2
        }
        
        model = XResNet1d101(
            input_channels=12,
            num_classes=metadata['num_classes'],
            stem_ks=7,
            block_ks=3,
            drop_rate=0.2
        ).to(DEVICE)
        
        # Load weights safely
        if load_model_checkpoint_safely(model, baseline_checkpoint, DEVICE):
            model.eval()
            
            # Get predictions
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for x, y in tqdm(test_loader_raw, desc="Getting predictions", leave=False):
                    x = x.to(DEVICE)
                    outputs = model(x)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    all_preds.append(probs)
                    all_labels.append(y.numpy())
            
            y_scores = np.vstack(all_preds)
            y_true = np.vstack(all_labels)
            
            # Use 0.5 as default threshold
            optimal_thresholds = np.ones(metadata['num_classes']) * 0.5
            y_pred = apply_thresholds(y_scores, optimal_thresholds)
            
            # Compute metrics
            metrics = compute_all_metrics(y_true, y_pred, y_scores)
            
            print(f"\nMetrics:")
            print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
            print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
            print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
            
            # Plot confusion matrices
            print(f"\nGenerating confusion matrices...")
            
            plot_combined_confusion(
                y_true, y_pred, metadata['classes'],
                save_path=os.path.join(ENSEMBLE_PATH, "confusion_combined_ResNet1D-Baseline.png"),
                title="Confusion Matrix"
            )
            
            plot_confusion_matrix_multiclass(
                y_true, y_pred, metadata['classes'],
                save_path=os.path.join(ENSEMBLE_PATH, "confusion_multiclass_ResNet1D-Baseline.png"),
                title="Multi-Class Confusion Matrix"
            )
            
            # Store results
            all_model_results['ResNet1D-Baseline'] = {
                'metrics': metrics,
                'y_scores': y_scores,
                'y_pred': y_pred,
                'thresholds': optimal_thresholds,
                'config': baseline_config
            }
            
            print(f"✓ ResNet1D-Baseline evaluated successfully")
        else:
            print(f"✗ Failed to load ResNet1D-Baseline")
    else:
        print(f"\n⚠️ ResNet1D baseline checkpoint not found at: {baseline_checkpoint}")
    
    # ========================================================================
    # PART 2: Evaluate CWT-based Models
    # ========================================================================
    
    # Find all CWT model result files
    result_files = [f for f in os.listdir(RESULTS_PATH) if f.startswith('results_') and f.endswith('.json')]
    
    if not result_files:
        print("⚠️ No CWT models found!")
    else:
        print(f"\nFound {len(result_files)} CWT models")
    
    for result_file in result_files:
        model_name = result_file.replace('results_', '').replace('.json', '')
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load results and config
        with open(os.path.join(RESULTS_PATH, result_file), 'r') as f:
            results = json.load(f)
        
        config = results['config']
        optimal_thresholds = np.array(results['optimal_thresholds'])
        
        # Load model
        checkpoint_path = os.path.join(RESULTS_PATH, f"best_{model_name}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️ Checkpoint not found: {checkpoint_path}")
            continue
        
        try:
            model = load_model_from_config(config, metadata['num_classes'])
            
            # Load checkpoint safely
            if not load_model_checkpoint_safely(model, checkpoint_path, DEVICE):
                print(f"  ✗ Skipping {model_name} due to loading errors")
                continue
            
            model.eval()
            
            # Create dataset with appropriate mode for this model
            dataset_mode = config['mode']
            
            test_dataset = CWTDataset(
                os.path.join(WAVELETS_PATH, 'test_scalograms.npy'),
                os.path.join(WAVELETS_PATH, 'test_phasograms.npy'),
                y_test,
                mode=dataset_mode
            )
            
            test_loader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True
            )
            
            # Get predictions
            y_scores, y_true = get_predictions(model, test_loader, config)
            y_pred = apply_thresholds(y_scores, optimal_thresholds)
            
            # Compute metrics
            metrics = compute_all_metrics(y_true, y_pred, y_scores)
            
            print(f"\nMetrics:")
            print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
            print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
            print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
            
            # Plot confusion matrices
            print(f"\nGenerating confusion matrices...")
            
            plot_confusion_matrix_multiclass(
                y_true, y_pred, metadata['classes'],
                save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{model_name}.png"),
                title=f"Multi-Class Confusion Matrix"
            )
            
            # Store results
            all_model_results[model_name] = {
                'metrics': metrics,
                'y_scores': y_scores,
                'y_pred': y_pred,
                'thresholds': optimal_thresholds,
                'config': config
            }
            
            print(f"✓ {model_name} evaluated successfully")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            print(f"  ✗ Skipping {model_name}")
            continue
    
    return all_model_results, y_true

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

@torch.no_grad()
def get_predictions(model, dataloader, config):
    """Get model predictions and probabilities"""
    model.eval()
    all_preds = []
    all_labels = []
    
    mode = config['mode']
    is_dual = (config['model'] == 'DualStream') or (config['mode'] == 'both')
    
    for batch in tqdm(dataloader, desc="Getting predictions", leave=False):
        # Handle different batch formats
        if isinstance(batch[0], tuple) or isinstance(batch[0], list):
            # Mode is 'both' - unpack tuple
            (x1, x2), y = batch
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            if is_dual:
                outputs = model(x1, x2)
            else:
                # Model expects single input, use appropriate one
                if mode == 'scalogram':
                    outputs = model(x1)
                elif mode == 'phasogram':
                    outputs = model(x2)
                elif mode == 'fusion':
                    # Concatenate for fusion models
                    x_fused = torch.cat([x1, x2], dim=1)
                    outputs = model(x_fused)
                else:
                    outputs = model(x1)
        else:
            # Single input (scalogram, phasogram, or fusion)
            x, y = batch
            x = x.to(DEVICE)
            outputs = model(x)
        
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(y.numpy())
    
    return np.vstack(all_preds), np.vstack(all_labels)

def apply_thresholds(y_scores, thresholds):
    """Apply class-wise thresholds"""
    y_pred = np.zeros_like(y_scores)
    for i in range(y_scores.shape[1]):
        y_pred[:, i] = (y_scores[:, i] > thresholds[i]).astype(int)
    
    # Ensure at least one prediction per sample
    for i, pred in enumerate(y_pred):
        if pred.sum() == 0:
            y_pred[i, np.argmax(y_scores[i])] = 1
    
    return y_pred

# ============================================================================
# CONFUSION MATRIX PLOTTING
# ============================================================================

def plot_confusion_matrix_per_class(y_true, y_pred, class_names, save_path=None, 
                                    title="Confusion Matrices"):
    """Plot separate confusion matrix for each class (binary: positive/negative)"""
    n_classes = y_true.shape[1]
    fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
    
    if n_classes == 1:
        axes = [axes]
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f'{class_name}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    plt.close()

def plot_confusion_matrix_multiclass(y_true, y_pred, class_names, save_path=None,
                                     title="Confusion Matrix - Multi-Class"):
    """
    Plot single confusion matrix treating as multi-class
    (convert multi-label to single class per sample)
    """
    # Convert multi-label to multi-class by taking argmax
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(class_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    plt.close()

def plot_combined_confusion(y_true, y_pred, class_names, save_path=None, 
                           title="Confusion Matrix"):
    """
    Plot both per-class binary and multi-class confusion matrices
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Left: Per-class binary
    n_classes = len(class_names)
    for i, class_name in enumerate(class_names):
        ax = plt.subplot(2, n_classes, i+1)
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                   cbar=False)
        ax.set_title(f'{class_name}', fontsize=10)
        if i == 0:
            ax.set_ylabel('True', fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9)
    
    # Bottom: Multi-class view
    ax_multi = plt.subplot(2, 1, 2)
    y_true_single = np.argmax(y_true, axis=1)
    y_pred_single = np.argmax(y_pred, axis=1)
    cm_multi = confusion_matrix(y_true_single, y_pred_single, labels=range(n_classes))
    
    sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens', ax=ax_multi,
                xticklabels=class_names, yticklabels=class_names)
    ax_multi.set_title('Multi-Class View (Argmax)', fontsize=12, fontweight='bold')
    ax_multi.set_xlabel('Predicted', fontsize=10)
    ax_multi.set_ylabel('True', fontsize=10)
    plt.setp(ax_multi.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    plt.close()

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_all_metrics(y_true, y_pred, y_scores):
    """Compute comprehensive metrics"""
    metrics = {
        'macro_auc': roc_auc_score(y_true, y_scores, average='macro'),
        'micro_auc': roc_auc_score(y_true, y_scores, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f_beta_macro': fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0),
        'per_class_auc': []
    }
    
    # Per-class metrics
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        except:
            auc = 0.0
        metrics['per_class_auc'].append(auc)
    
    return metrics

# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def evaluate_all_models(metadata, X_test_raw, y_test):
    """Evaluate all trained models and generate confusion matrices"""
    
    print("\n[1/3] Loading all trained models...")
    
    all_model_results = {}
    y_true = y_test
    
    # ========================================================================
    # PART 1: Evaluate ResNet1D Baseline
    # ========================================================================
    
    baseline_checkpoint = os.path.join(BASELINE_RESULTS_PATH, 'best_xresnet1d101.pth')
    
    if os.path.exists(baseline_checkpoint):
        print(f"\n{'='*60}")
        print(f"Evaluating: ResNet1D-Baseline")
        print(f"{'='*60}")
        
        # Create dataset for raw signals
        test_dataset_raw = ECGDataset(X_test_raw, y_test)
        test_loader_raw = DataLoader(
            test_dataset_raw, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # Load model
        baseline_config = {
            'model': 'XResNet1d101',
            'mode': 'raw',
            'stem_ks': 7,
            'block_ks': 3,
            'drop_rate': 0.2
        }
        
        model = XResNet1d101(
            input_channels=12,
            num_classes=metadata['num_classes'],
            stem_ks=7,
            block_ks=3,
            drop_rate=0.2
        ).to(DEVICE)
        
        # Load weights
        state_dict = torch.load(baseline_checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Get predictions
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in tqdm(test_loader_raw, desc="Getting predictions", leave=False):
                x = x.to(DEVICE)
                outputs = model(x)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(y.numpy())
        
        y_scores = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        
        # Use 0.5 as default threshold (or load from saved thresholds if available)
        optimal_thresholds = np.ones(metadata['num_classes']) * 0.5
        y_pred = apply_thresholds(y_scores, optimal_thresholds)
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred, y_scores)
        
        print(f"\nMetrics:")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
        print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
        
        # Plot confusion matrices
        print(f"\nGenerating confusion matrices...")
        
        plot_combined_confusion(
            y_true, y_pred, metadata['classes'],
            save_path=os.path.join(ENSEMBLE_PATH, "confusion_combined_ResNet1D-Baseline.png"),
            title="Confusion Matrix"
        )
        
        plot_confusion_matrix_multiclass(
            y_true, y_pred, metadata['classes'],
            save_path=os.path.join(ENSEMBLE_PATH, "confusion_multiclass_ResNet1D-Baseline.png"),
            title="Multi-Class Confusion Matrix"
        )
        
        # Store results
        all_model_results['ResNet1D-Baseline'] = {
            'metrics': metrics,
            'y_scores': y_scores,
            'y_pred': y_pred,
            'thresholds': optimal_thresholds,
            'config': baseline_config
        }
        
        print(f"✓ ResNet1D-Baseline evaluated successfully")
    else:
        print(f"\n⚠️ ResNet1D baseline checkpoint not found at: {baseline_checkpoint}")
    
    # ========================================================================
    # PART 2: Evaluate CWT-based Models
    # ========================================================================
    
    # Find all CWT model result files
    result_files = [f for f in os.listdir(RESULTS_PATH) if f.startswith('results_') and f.endswith('.json')]
    
    if not result_files:
        print("⚠️ No CWT models found!")
    else:
        print(f"\nFound {len(result_files)} CWT models")
    
    for result_file in result_files:
        model_name = result_file.replace('results_', '').replace('.json', '')
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load results and config
        with open(os.path.join(RESULTS_PATH, result_file), 'r') as f:
            results = json.load(f)
        
        config = results['config']
        optimal_thresholds = np.array(results['optimal_thresholds'])
        
        # Load model
        checkpoint_path = os.path.join(RESULTS_PATH, f"best_{model_name}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️ Checkpoint not found: {checkpoint_path}")
            continue
        
        model = load_model_from_config(config, metadata['num_classes'])
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dataset with appropriate mode for this model
        dataset_mode = config['mode']
        
        test_dataset = CWTDataset(
            os.path.join(WAVELETS_PATH, 'test_scalograms.npy'),
            os.path.join(WAVELETS_PATH, 'test_phasograms.npy'),
            y_test,
            mode=dataset_mode
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # Get predictions
        y_scores, y_true = get_predictions(model, test_loader, config)
        y_pred = apply_thresholds(y_scores, optimal_thresholds)
        
        # Compute metrics
        metrics = compute_all_metrics(y_true, y_pred, y_scores)
        
        print(f"\nMetrics:")
        print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
        print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
        
        # Plot confusion matrices
        print(f"\nGenerating confusion matrices...")
        
        # plot_combined_confusion(
        #     y_true, y_pred, metadata['classes'],
        #     save_path=os.path.join(ENSEMBLE_PATH, f"confusion_combined_{model_name}.png"),
        #     title=f"{model_name} - Confusion Matrix"
        # )
        
        plot_confusion_matrix_multiclass(
            y_true, y_pred, metadata['classes'],
            save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{model_name}.png"),
            title=f"Multi-Class Confusion Matrix"
        )
        
        # Store results
        all_model_results[model_name] = {
            'metrics': metrics,
            'y_scores': y_scores,
            'y_pred': y_pred,
            'thresholds': optimal_thresholds,
            'config': config
        }
    
    return all_model_results, y_true

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

def create_ensemble(model_results, y_true, metadata, method='average', top_k=None):
    """Create ensemble predictions"""
    
    print(f"\n[2/3] Creating ensemble (method={method}, top_k={top_k})...")
    
    # Select models
    if top_k:
        # Sort by AUC and take top k
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1]['metrics']['macro_auc'],
            reverse=True
        )[:top_k]
        selected_names = [name for name, _ in sorted_models]
        print(f"\nTop {top_k} models selected:")
        for name in selected_names:
            auc = model_results[name]['metrics']['macro_auc']
            print(f"  - {name}: AUC={auc:.4f}")
    else:
        selected_names = list(model_results.keys())
        print(f"\nUsing all {len(selected_names)} models")
    
    # Collect scores
    all_scores = [model_results[name]['y_scores'] for name in selected_names]
    all_thresholds = [model_results[name]['thresholds'] for name in selected_names]
    
    # Ensemble predictions
    if method == 'average':
        ensemble_scores = np.mean(all_scores, axis=0)
        ensemble_thresholds = np.mean(all_thresholds, axis=0)
    elif method == 'weighted':
        # Weight by AUC
        weights = [model_results[name]['metrics']['macro_auc'] for name in selected_names]
        weights = np.array(weights) / sum(weights)
        ensemble_scores = np.average(all_scores, axis=0, weights=weights)
        ensemble_thresholds = np.average(all_thresholds, axis=0, weights=weights)
    elif method == 'max':
        ensemble_scores = np.max(all_scores, axis=0)
        ensemble_thresholds = np.mean(all_thresholds, axis=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Apply thresholds
    ensemble_pred = apply_thresholds(ensemble_scores, ensemble_thresholds)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, ensemble_pred, ensemble_scores)
    
    print(f"\nEnsemble Metrics:")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
    
    # Plot confusion matrices
    ensemble_name = f"ensemble_{method}" + (f"_top{top_k}" if top_k else "_all")
    
    plot_combined_confusion(
        y_true, ensemble_pred, metadata['classes'],
        save_path=os.path.join(ENSEMBLE_PATH, f"confusion_combined_{ensemble_name}.png"),
        title=f"Ensemble ({method.title()}) - Confusion Matrix"
    )
    
    plot_confusion_matrix_multiclass(
        y_true, ensemble_pred, metadata['classes'],
        save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{ensemble_name}.png"),
        title=f"Ensemble ({method.title()}) - Multi-Class"
    )
    
    return metrics, ensemble_name

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_model_comparison(model_results, ensemble_results, metadata):
    """Plot comparison of all models and ensembles"""
    
    print("\n[3/3] Creating comparison visualizations...")
    
    # Prepare data
    names = list(model_results.keys()) + [name for name, _ in ensemble_results]
    aucs = [model_results[n]['metrics']['macro_auc'] for n in model_results.keys()] + \
           [metrics['macro_auc'] for _, metrics in ensemble_results]
    f1s = [model_results[n]['metrics']['f1_macro'] for n in model_results.keys()] + \
          [metrics['f1_macro'] for _, metrics in ensemble_results]
    
    # Sort by AUC
    sorted_indices = np.argsort(aucs)[::-1]
    names = [names[i] for i in sorted_indices]
    aucs = [aucs[i] for i in sorted_indices]
    f1s = [f1s[i] for i in sorted_indices]
    
    # Plot AUC comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#1f77b4'] * len(model_results) + ['#ff7f0e'] * len(ensemble_results)
    colors = [colors[i] for i in sorted_indices]
    
    ax1.barh(range(len(names)), aucs, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel('Macro AUC', fontsize=11, fontweight='bold')
    ax1.set_title('Model Performance - AUC', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=np.mean(aucs), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax1.legend()
    
    # Add value labels
    for i, v in enumerate(aucs):
        ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    
    # Plot F1 comparison
    ax2.barh(range(len(names)), f1s, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Macro F1', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance - F1', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=np.mean(f1s), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax2.legend()
    
    # Add value labels
    for i, v in enumerate(f1s):
        ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ENSEMBLE_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: model_comparison.png")
    plt.close()
    
    # Per-class AUC heatmap
    per_class_data = []
    for name in model_results.keys():
        per_class_data.append(model_results[name]['metrics']['per_class_auc'])
    
    if per_class_data:
        plt.figure(figsize=(10, max(6, len(model_results) * 0.4)))
        sns.heatmap(
            per_class_data,
            xticklabels=metadata['classes'],
            yticklabels=list(model_results.keys()),
            annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'AUC'}
        )
        plt.title('Per-Class AUC Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=11)
        plt.ylabel('Model', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(ENSEMBLE_PATH, 'per_class_auc_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: per_class_auc_heatmap.png")
        plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n[Step 1] Loading metadata and test data...")
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
    
    X_val = X[Y.strat_fold == 9]
    
    X_test = X[Y.strat_fold == 10]
    y_test = y[Y.strat_fold == 10]
    
    
    # Step 4: Preprocess
    print("\n[4/8] Preprocessing signals...")
    X_train, X_val, X_test, scaler = preprocess_signals(X_train, X_val, X_test)
    # Load metadata
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset info:")
    print(f"  Classes: {metadata['num_classes']} - {metadata['classes']}")
    print(f"  Test samples: {metadata['test_size']}")
    
    # Load test data
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    # Load raw test signals for ResNet1D baseline
    
    print(f"  Raw signals shape: {X_test.shape}")
    
    # Evaluate all models (CWT + ResNet1D baseline)
    model_results, y_true = evaluate_all_models(metadata, X_test, y_test)
    
    if not model_results:
        print("\n❌ No models to evaluate!")
        return
    
    print(f"\n✓ Successfully evaluated {len(model_results)} models")
    
    # Create ensembles
    ensemble_results = []
    
    # Average ensemble (all models)
    metrics, name = create_ensemble(model_results, y_true, metadata, method='average')
    ensemble_results.append((name, metrics))
    
    # Weighted ensemble (all models)
    metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted')
    ensemble_results.append((name, metrics))
    
    # Top-3 average ensemble
    if len(model_results) >= 3:
        metrics, name = create_ensemble(model_results, y_true, metadata, method='average', top_k=3)
        ensemble_results.append((name, metrics))
    
    # Top-3 weighted ensemble
    if len(model_results) >= 3:
        metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted', top_k=3)
        ensemble_results.append((name, metrics))
    
    # Top-5 ensembles if we have enough models
    if len(model_results) >= 5:
        metrics, name = create_ensemble(model_results, y_true, metadata, method='average', top_k=5)
        ensemble_results.append((name, metrics))
        
        metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted', top_k=5)
        ensemble_results.append((name, metrics))
    
    # Create comparison visualizations
    plot_model_comparison(model_results, ensemble_results, metadata)
    
    # Save summary
    print("\n" + "="*80)
    print("FINAL SUMMARY (Including ResNet1D Baseline)")
    print("="*80)
    
    summary = {
        'individual_models': {
            name: {
                'macro_auc': results['metrics']['macro_auc'],
                'f1_macro': results['metrics']['f1_macro'],
                'f_beta_macro': results['metrics']['f_beta_macro']
            }
            for name, results in model_results.items()
        },
        'ensembles': {
            name: {
                'macro_auc': metrics['macro_auc'],
                'f1_macro': metrics['f1_macro'],
                'f_beta_macro': metrics['f_beta_macro']
            }
            for name, metrics in ensemble_results
        }
    }
    
    with open(os.path.join(ENSEMBLE_PATH, 'complete_results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    print(f"\n{'Model':<40} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
    print("-" * 80)
    
    # Individual models (sorted by AUC)
    for name, results in sorted(model_results.items(), 
                                key=lambda x: x[1]['metrics']['macro_auc'], 
                                reverse=True):
        m = results['metrics']
        marker = "🏆 " if name == "ResNet1D-Baseline" else "   "
        print(f"{marker}{name:<37} | {m['macro_auc']:.4f}   | {m['f1_macro']:.4f}   | {m['f_beta_macro']:.4f}")
    
    print("-" * 80)
    
    # Ensembles
    for name, metrics in ensemble_results:
        print(f"🎯 {name:<37} | {metrics['macro_auc']:.4f}   | {metrics['f1_macro']:.4f}   | {metrics['f_beta_macro']:.4f}")
    
    # Find best model
    best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['macro_auc'])
    best_ensemble = max(ensemble_results, key=lambda x: x[1]['macro_auc'])
    
    print("\n" + "="*80)
    print("🏅 BEST PERFORMERS")
    print("="*80)
    print(f"Best Individual Model: {best_model[0]} (AUC: {best_model[1]['metrics']['macro_auc']:.4f})")
    print(f"Best Ensemble:         {best_ensemble[0]} (AUC: {best_ensemble[1]['macro_auc']:.4f})")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {ENSEMBLE_PATH}")
    print("\nGenerated files:")
    print("  - Confusion matrices for all models (including ResNet1D)")
    print("  - Ensemble confusion matrices")
    print("  - Model comparison plots")
    print("  - Per-class AUC heatmap")
    print("  - Complete results summary (JSON)")
    print("\n📊 Key Insights:")
    print(f"  • Total models evaluated: {len(model_results)}")
    print(f"  • Ensemble combinations: {len(ensemble_results)}")
    print(f"  • ResNet1D baseline included: {'✓' if 'ResNet1D-Baseline' in model_results else '✗'}")

if __name__ == '__main__':
    main()