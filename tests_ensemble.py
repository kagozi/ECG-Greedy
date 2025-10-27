# # ============================================================================
# # CONFUSION MATRIX PLOTTING AND MODEL ENSEMBLING
# # ============================================================================
# # Run this after training all models to:
# # 1. Generate confusion matrices for all trained models
# # 2. Create ensemble predictions
# # 3. Compare all models visually

# import os
# import json
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import (
#     confusion_matrix, roc_auc_score, f1_score, 
#     fbeta_score, classification_report
# )
# from tqdm import tqdm

# # Import your models
# from models import (
#     CWT2DCNN, DualStreamCNN, ViTECG, SwinTransformerECG,
#     SwinTransformerEarlyFusion, SwinTransformerLateFusion,
#     EfficientNetECG, ViTFusionECG, EfficientNetFusionECG
# )

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
# WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/'
# RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/results/'
# ENSEMBLE_PATH = os.path.join(RESULTS_PATH, 'ensemble_results/')
# BATCH_SIZE = 8
# NUM_WORKERS = 4
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.makedirs(ENSEMBLE_PATH, exist_ok=True)

# print("="*80)
# print("CONFUSION MATRICES & ENSEMBLE ANALYSIS")
# print("="*80)
# print(f"Device: {DEVICE}")

# # ============================================================================
# # DATASET CLASS (Same as training)
# # ============================================================================

# class CWTDataset(Dataset):
#     """Memory-efficient dataset for CWT data"""
    
#     def __init__(self, scalo_path, phaso_path, labels, mode='scalogram'):
#         self.scalograms = np.load(scalo_path, mmap_mode='r')
#         self.phasograms = np.load(phaso_path, mmap_mode='r')
#         self.labels = torch.FloatTensor(labels)
#         self.mode = mode
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
#         phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
#         label = self.labels[idx]
        
#         if self.mode == 'scalogram':
#             return scalo, label
#         elif self.mode == 'phasogram':
#             return phaso, label
#         elif self.mode == 'both':
#             return (scalo, phaso), label
#         elif self.mode == 'fusion':
#             fused = torch.cat([scalo, phaso], dim=0)
#             return fused, label

# # ============================================================================
# # MODEL LOADING UTILITIES
# # ============================================================================

# def load_model_from_config(config, num_classes):
#     """Load model architecture based on config"""
#     mode = config['mode']
#     model_type = config['model']
#     adapter_strategy = config.get('adapter', 'learned')
    
#     if model_type == 'DualStream':
#         model = DualStreamCNN(num_classes=num_classes, num_channels=12)
#     elif model_type == 'CWT2DCNN':
#         num_ch = 24 if mode == 'fusion' else 12
#         model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
#     elif model_type == 'ViTECG':
#         model = ViTECG(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
#     elif model_type == 'SwinTransformerECG':
#         model = SwinTransformerECG(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
#     elif model_type == 'SwinTransformerEarlyFusion':
#         model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=False)
#     elif model_type == 'SwinTransformerLateFusion':
#         model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
#     elif model_type == 'EfficientNetECG':
#         model = EfficientNetECG(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
#     elif model_type == 'ViTFusionECG':
#         model = ViTFusionECG(num_classes=num_classes, pretrained=False)
#     elif model_type == 'EfficientNetFusionECG':
#         model = EfficientNetFusionECG(num_classes=num_classes, pretrained=False)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
    
#     return model.to(DEVICE)

# # ============================================================================
# # PREDICTION FUNCTIONS
# # ============================================================================

# @torch.no_grad()
# def get_predictions(model, dataloader, config):
#     """Get model predictions and probabilities"""
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     mode = config['mode']
#     is_dual = (config['model'] == 'DualStream') or (config['mode'] == 'both')
    
#     for batch in tqdm(dataloader, desc="Getting predictions", leave=False):
#         # Handle different batch formats
#         if isinstance(batch[0], tuple) or isinstance(batch[0], list):
#             # Mode is 'both' - unpack tuple
#             (x1, x2), y = batch
#             x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
#             if is_dual:
#                 outputs = model(x1, x2)
#             else:
#                 # Model expects single input, use appropriate one
#                 if mode == 'scalogram':
#                     outputs = model(x1)
#                 elif mode == 'phasogram':
#                     outputs = model(x2)
#                 elif mode == 'fusion':
#                     # Concatenate for fusion models
#                     x_fused = torch.cat([x1, x2], dim=1)
#                     outputs = model(x_fused)
#                 else:
#                     outputs = model(x1)
#         else:
#             # Single input (scalogram, phasogram, or fusion)
#             x, y = batch
#             x = x.to(DEVICE)
#             outputs = model(x)
        
#         probs = torch.sigmoid(outputs).cpu().numpy()
#         all_preds.append(probs)
#         all_labels.append(y.numpy())
    
#     return np.vstack(all_preds), np.vstack(all_labels)

# def apply_thresholds(y_scores, thresholds):
#     """Apply class-wise thresholds"""
#     y_pred = np.zeros_like(y_scores)
#     for i in range(y_scores.shape[1]):
#         y_pred[:, i] = (y_scores[:, i] > thresholds[i]).astype(int)
    
#     # Ensure at least one prediction per sample
#     for i, pred in enumerate(y_pred):
#         if pred.sum() == 0:
#             y_pred[i, np.argmax(y_scores[i])] = 1
    
#     return y_pred

# # ============================================================================
# # CONFUSION MATRIX PLOTTING
# # ============================================================================

# def plot_confusion_matrix_per_class(y_true, y_pred, class_names, save_path=None, 
#                                     title="Confusion Matrices"):
#     """Plot separate confusion matrix for each class (binary: positive/negative)"""
#     n_classes = y_true.shape[1]
#     fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 4))
    
#     if n_classes == 1:
#         axes = [axes]
    
#     for i, (ax, class_name) in enumerate(zip(axes, class_names)):
#         cm = confusion_matrix(y_true[:, i], y_pred[:, i])
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
#                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
#         ax.set_title(f'{class_name}')
#         ax.set_ylabel('True')
#         ax.set_xlabel('Predicted')
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"  ✓ Saved: {save_path}")
#     plt.close()

# def plot_confusion_matrix_multiclass(y_true, y_pred, class_names, save_path=None,
#                                      title="Confusion Matrix - Multi-Class"):
#     """
#     Plot single confusion matrix treating as multi-class
#     (convert multi-label to single class per sample)
#     """
#     # Convert multi-label to multi-class by taking argmax
#     y_true_single = np.argmax(y_true, axis=1)
#     y_pred_single = np.argmax(y_pred, axis=1)
    
#     cm = confusion_matrix(y_true_single, y_pred_single, labels=range(len(class_names)))
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names,
#                 cbar_kws={'shrink': 0.8})
#     plt.xlabel("Predicted", fontsize=12)
#     plt.ylabel("True", fontsize=12)
#     plt.title(title, fontsize=14, fontweight='bold')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"  ✓ Saved: {save_path}")
#     plt.close()

# def plot_combined_confusion(y_true, y_pred, class_names, save_path=None, 
#                            title="Confusion Matrix"):
#     """
#     Plot both per-class binary and multi-class confusion matrices
#     """
#     fig = plt.figure(figsize=(16, 6))
    
#     # Left: Per-class binary
#     n_classes = len(class_names)
#     for i, class_name in enumerate(class_names):
#         ax = plt.subplot(2, n_classes, i+1)
#         cm = confusion_matrix(y_true[:, i], y_pred[:, i])
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
#                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
#                    cbar=False)
#         ax.set_title(f'{class_name}', fontsize=10)
#         if i == 0:
#             ax.set_ylabel('True', fontsize=9)
#         ax.set_xlabel('Predicted', fontsize=9)
    
#     # Bottom: Multi-class view
#     ax_multi = plt.subplot(2, 1, 2)
#     y_true_single = np.argmax(y_true, axis=1)
#     y_pred_single = np.argmax(y_pred, axis=1)
#     cm_multi = confusion_matrix(y_true_single, y_pred_single, labels=range(n_classes))
    
#     sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Greens', ax=ax_multi,
#                 xticklabels=class_names, yticklabels=class_names)
#     ax_multi.set_title('Multi-Class View (Argmax)', fontsize=12, fontweight='bold')
#     ax_multi.set_xlabel('Predicted', fontsize=10)
#     ax_multi.set_ylabel('True', fontsize=10)
#     plt.setp(ax_multi.get_xticklabels(), rotation=45, ha='right')
    
#     plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"  ✓ Saved: {save_path}")
#     plt.close()

# # ============================================================================
# # EVALUATION METRICS
# # ============================================================================

# def compute_all_metrics(y_true, y_pred, y_scores):
#     """Compute comprehensive metrics"""
#     metrics = {
#         'macro_auc': roc_auc_score(y_true, y_scores, average='macro'),
#         'micro_auc': roc_auc_score(y_true, y_scores, average='micro'),
#         'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
#         'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
#         'f_beta_macro': fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0),
#         'per_class_auc': []
#     }
    
#     # Per-class metrics
#     for i in range(y_true.shape[1]):
#         try:
#             auc = roc_auc_score(y_true[:, i], y_scores[:, i])
#         except:
#             auc = 0.0
#         metrics['per_class_auc'].append(auc)
    
#     return metrics

# # ============================================================================
# # MAIN EVALUATION PIPELINE
# # ============================================================================

# def evaluate_all_models(metadata):
#     """Evaluate all trained models and generate confusion matrices"""
    
#     print("\n[1/3] Loading all trained models...")
    
#     # Find all result files
#     result_files = [f for f in os.listdir(RESULTS_PATH) if f.startswith('results_') and f.endswith('.json')]
    
#     if not result_files:
#         print("❌ No trained models found!")
#         return None
    
#     print(f"Found {len(result_files)} trained models")
    
#     all_model_results = {}
#     y_true = None
    
#     for result_file in result_files:
#         model_name = result_file.replace('results_', '').replace('.json', '')
        
#         print(f"\n{'='*60}")
#         print(f"Evaluating: {model_name}")
#         print(f"{'='*60}")
        
#         # Load results and config
#         with open(os.path.join(RESULTS_PATH, result_file), 'r') as f:
#             results = json.load(f)
        
#         config = results['config']
#         optimal_thresholds = np.array(results['optimal_thresholds'])
        
#         # Load model
#         checkpoint_path = os.path.join(RESULTS_PATH, f"best_{model_name}.pth")
#         if not os.path.exists(checkpoint_path):
#             print(f"  ⚠️ Checkpoint not found: {checkpoint_path}")
#             continue
        
#         model = load_model_from_config(config, metadata['num_classes'])
#         checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.eval()
        
#         # Create dataset with appropriate mode for this model
#         y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
        
#         # Determine the mode needed for this model
#         dataset_mode = config['mode']
        
#         test_dataset = CWTDataset(
#             os.path.join(WAVELETS_PATH, 'test_scalograms.npy'),
#             os.path.join(WAVELETS_PATH, 'test_phasograms.npy'),
#             y_test,
#             mode=dataset_mode
#         )
        
#         test_loader = DataLoader(
#             test_dataset, batch_size=BATCH_SIZE, shuffle=False,
#             num_workers=NUM_WORKERS, pin_memory=True
#         )
        
#         # Get predictions
#         y_scores, y_true = get_predictions(model, test_loader, config)
#         y_pred = apply_thresholds(y_scores, optimal_thresholds)
        
#         # Compute metrics
#         metrics = compute_all_metrics(y_true, y_pred, y_scores)
        
#         print(f"\nMetrics:")
#         print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
#         print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
#         print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
        
#         # Plot confusion matrices
#         print(f"\nGenerating confusion matrices...")
        
#         # Combined view
#         plot_combined_confusion(
#             y_true, y_pred, metadata['classes'],
#             save_path=os.path.join(ENSEMBLE_PATH, f"confusion_combined_{model_name}.png"),
#             title=f"{model_name} - Confusion Matrix"
#         )
        
#         # Multi-class only
#         plot_confusion_matrix_multiclass(
#             y_true, y_pred, metadata['classes'],
#             save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{model_name}.png"),
#             title=f"{model_name} - Multi-Class Confusion Matrix"
#         )
        
#         # Store results
#         all_model_results[model_name] = {
#             'metrics': metrics,
#             'y_scores': y_scores,
#             'y_pred': y_pred,
#             'thresholds': optimal_thresholds,
#             'config': config
#         }
    
#     return all_model_results, y_true

# # ============================================================================
# # ENSEMBLE METHODS
# # ============================================================================

# def create_ensemble(model_results, y_true, metadata, method='average', top_k=None):
#     """Create ensemble predictions"""
    
#     print(f"\n[2/3] Creating ensemble (method={method}, top_k={top_k})...")
    
#     # Select models
#     if top_k:
#         # Sort by AUC and take top k
#         sorted_models = sorted(
#             model_results.items(),
#             key=lambda x: x[1]['metrics']['macro_auc'],
#             reverse=True
#         )[:top_k]
#         selected_names = [name for name, _ in sorted_models]
#         print(f"\nTop {top_k} models selected:")
#         for name in selected_names:
#             auc = model_results[name]['metrics']['macro_auc']
#             print(f"  - {name}: AUC={auc:.4f}")
#     else:
#         selected_names = list(model_results.keys())
#         print(f"\nUsing all {len(selected_names)} models")
    
#     # Collect scores
#     all_scores = [model_results[name]['y_scores'] for name in selected_names]
#     all_thresholds = [model_results[name]['thresholds'] for name in selected_names]
    
#     # Ensemble predictions
#     if method == 'average':
#         ensemble_scores = np.mean(all_scores, axis=0)
#         ensemble_thresholds = np.mean(all_thresholds, axis=0)
#     elif method == 'weighted':
#         # Weight by AUC
#         weights = [model_results[name]['metrics']['macro_auc'] for name in selected_names]
#         weights = np.array(weights) / sum(weights)
#         ensemble_scores = np.average(all_scores, axis=0, weights=weights)
#         ensemble_thresholds = np.average(all_thresholds, axis=0, weights=weights)
#     elif method == 'max':
#         ensemble_scores = np.max(all_scores, axis=0)
#         ensemble_thresholds = np.mean(all_thresholds, axis=0)
#     else:
#         raise ValueError(f"Unknown ensemble method: {method}")
    
#     # Apply thresholds
#     ensemble_pred = apply_thresholds(ensemble_scores, ensemble_thresholds)
    
#     # Compute metrics
#     metrics = compute_all_metrics(y_true, ensemble_pred, ensemble_scores)
    
#     print(f"\nEnsemble Metrics:")
#     print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
#     print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
#     print(f"  F-beta:    {metrics['f_beta_macro']:.4f}")
    
#     # Plot confusion matrices
#     ensemble_name = f"ensemble_{method}" + (f"_top{top_k}" if top_k else "_all")
    
#     plot_combined_confusion(
#         y_true, ensemble_pred, metadata['classes'],
#         save_path=os.path.join(ENSEMBLE_PATH, f"confusion_combined_{ensemble_name}.png"),
#         title=f"Ensemble ({method.title()}) - Confusion Matrix"
#     )
    
#     plot_confusion_matrix_multiclass(
#         y_true, ensemble_pred, metadata['classes'],
#         save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{ensemble_name}.png"),
#         title=f"Ensemble ({method.title()}) - Multi-Class"
#     )
    
#     return metrics, ensemble_name

# # ============================================================================
# # VISUALIZATION
# # ============================================================================

# def plot_model_comparison(model_results, ensemble_results, metadata):
#     """Plot comparison of all models and ensembles"""
    
#     print("\n[3/3] Creating comparison visualizations...")
    
#     # Prepare data
#     names = list(model_results.keys()) + [name for name, _ in ensemble_results]
#     aucs = [model_results[n]['metrics']['macro_auc'] for n in model_results.keys()] + \
#            [metrics['macro_auc'] for _, metrics in ensemble_results]
#     f1s = [model_results[n]['metrics']['f1_macro'] for n in model_results.keys()] + \
#           [metrics['f1_macro'] for _, metrics in ensemble_results]
    
#     # Sort by AUC
#     sorted_indices = np.argsort(aucs)[::-1]
#     names = [names[i] for i in sorted_indices]
#     aucs = [aucs[i] for i in sorted_indices]
#     f1s = [f1s[i] for i in sorted_indices]
    
#     # Plot AUC comparison
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     colors = ['#1f77b4'] * len(model_results) + ['#ff7f0e'] * len(ensemble_results)
#     colors = [colors[i] for i in sorted_indices]
    
#     ax1.barh(range(len(names)), aucs, color=colors, alpha=0.7)
#     ax1.set_yticks(range(len(names)))
#     ax1.set_yticklabels(names, fontsize=9)
#     ax1.set_xlabel('Macro AUC', fontsize=11, fontweight='bold')
#     ax1.set_title('Model Performance - AUC', fontsize=12, fontweight='bold')
#     ax1.grid(axis='x', alpha=0.3)
#     ax1.axvline(x=np.mean(aucs), color='red', linestyle='--', alpha=0.5, label='Mean')
#     ax1.legend()
    
#     # Add value labels
#     for i, v in enumerate(aucs):
#         ax1.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    
#     # Plot F1 comparison
#     ax2.barh(range(len(names)), f1s, color=colors, alpha=0.7)
#     ax2.set_yticks(range(len(names)))
#     ax2.set_yticklabels(names, fontsize=9)
#     ax2.set_xlabel('Macro F1', fontsize=11, fontweight='bold')
#     ax2.set_title('Model Performance - F1', fontsize=12, fontweight='bold')
#     ax2.grid(axis='x', alpha=0.3)
#     ax2.axvline(x=np.mean(f1s), color='red', linestyle='--', alpha=0.5, label='Mean')
#     ax2.legend()
    
#     # Add value labels
#     for i, v in enumerate(f1s):
#         ax2.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(ENSEMBLE_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
#     print(f"  ✓ Saved: model_comparison.png")
#     plt.close()
    
#     # Per-class AUC heatmap
#     per_class_data = []
#     for name in model_results.keys():
#         per_class_data.append(model_results[name]['metrics']['per_class_auc'])
    
#     if per_class_data:
#         plt.figure(figsize=(10, max(6, len(model_results) * 0.4)))
#         sns.heatmap(
#             per_class_data,
#             xticklabels=metadata['classes'],
#             yticklabels=list(model_results.keys()),
#             annot=True, fmt='.3f', cmap='YlOrRd',
#             cbar_kws={'label': 'AUC'}
#         )
#         plt.title('Per-Class AUC Heatmap', fontsize=14, fontweight='bold')
#         plt.xlabel('Class', fontsize=11)
#         plt.ylabel('Model', fontsize=11)
#         plt.tight_layout()
#         plt.savefig(os.path.join(ENSEMBLE_PATH, 'per_class_auc_heatmap.png'), dpi=300, bbox_inches='tight')
#         print(f"  ✓ Saved: per_class_auc_heatmap.png")
#         plt.close()

# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     print("\n[Step 1] Loading metadata and test data...")
    
#     # Load metadata
#     with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
#         metadata = pickle.load(f)
    
#     print(f"Dataset info:")
#     print(f"  Classes: {metadata['num_classes']} - {metadata['classes']}")
#     print(f"  Test samples: {metadata['test_size']}")
    
#     # Evaluate all models (creates dataloaders internally per model)
#     model_results, y_true = evaluate_all_models(metadata)
    
#     if not model_results:
#         print("\n❌ No models to evaluate!")
#         return
    
#     print(f"\n✓ Successfully evaluated {len(model_results)} models")
    
#     # Create ensembles
#     ensemble_results = []
    
#     # Average ensemble (all models)
#     metrics, name = create_ensemble(model_results, y_true, metadata, method='average')
#     ensemble_results.append((name, metrics))
    
#     # Weighted ensemble (all models)
#     metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted')
#     ensemble_results.append((name, metrics))
    
#     # Top-3 average ensemble
#     if len(model_results) >= 3:
#         metrics, name = create_ensemble(model_results, y_true, metadata, method='average', top_k=3)
#         ensemble_results.append((name, metrics))
    
#     # Top-3 weighted ensemble
#     if len(model_results) >= 3:
#         metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted', top_k=3)
#         ensemble_results.append((name, metrics))
    
#     # Top-5 ensembles if we have enough models
#     if len(model_results) >= 5:
#         metrics, name = create_ensemble(model_results, y_true, metadata, method='average', top_k=5)
#         ensemble_results.append((name, metrics))
        
#         metrics, name = create_ensemble(model_results, y_true, metadata, method='weighted', top_k=5)
#         ensemble_results.append((name, metrics))
    
#     # Create comparison visualizations
#     plot_model_comparison(model_results, ensemble_results, metadata)
    
#     # Save summary
#     print("\n" + "="*80)
#     print("FINAL SUMMARY")
#     print("="*80)
    
#     summary = {
#         'individual_models': {
#             name: {
#                 'macro_auc': results['metrics']['macro_auc'],
#                 'f1_macro': results['metrics']['f1_macro'],
#                 'f_beta_macro': results['metrics']['f_beta_macro']
#             }
#             for name, results in model_results.items()
#         },
#         'ensembles': {
#             name: {
#                 'macro_auc': metrics['macro_auc'],
#                 'f1_macro': metrics['f1_macro'],
#                 'f_beta_macro': metrics['f_beta_macro']
#             }
#             for name, metrics in ensemble_results
#         }
#     }
    
#     with open(os.path.join(ENSEMBLE_PATH, 'complete_results_summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)
    
#     # Print summary table
#     print(f"\n{'Model':<40} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
#     print("-" * 80)
    
#     # Individual models
#     for name, results in sorted(model_results.items(), 
#                                 key=lambda x: x[1]['metrics']['macro_auc'], 
#                                 reverse=True):
#         m = results['metrics']
#         print(f"{name:<40} | {m['macro_auc']:.4f}   | {m['f1_macro']:.4f}   | {m['f_beta_macro']:.4f}")
    
#     print("-" * 80)
    
#     # Ensembles
#     for name, metrics in ensemble_results:
#         print(f"{name:<40} | {metrics['macro_auc']:.4f}   | {metrics['f1_macro']:.4f}   | {metrics['f_beta_macro']:.4f}")
    
#     print("\n" + "="*80)
#     print("✓ ANALYSIS COMPLETE!")
#     print("="*80)
#     print(f"\nAll results saved to: {ENSEMBLE_PATH}")
#     print("\nGenerated files:")
#     print("  - Confusion matrices for all models")
#     print("  - Ensemble confusion matrices")
#     print("  - Model comparison plots")
#     print("  - Per-class AUC heatmap")
#     print("  - Complete results summary (JSON)")

# if __name__ == '__main__':
#     main()

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
from models import (
    CWT2DCNN, DualStreamCNN, ViTECG, SwinTransformerECG,
    SwinTransformerEarlyFusion, SwinTransformerLateFusion,
    EfficientNetECG, ViTFusionECG, EfficientNetFusionECG
)

# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
WAVELETS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/'
RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/cwt/processed_wavelets/results/'
BASELINE_RESULTS_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/results/'
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
# RESNET1D BASELINE MODEL ARCHITECTURE
# ============================================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class BasicBlock1d(nn.Module):
    """Basic ResNet block for 1D signals"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, drop_rate=0.0):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.drop_rate = drop_rate
        
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
        
        out = drop_path(out, self.drop_rate, self.training)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class XResNet1d101(nn.Module):
    """XResNet1d-101 architecture for baseline"""
    
    def __init__(self, input_channels=12, num_classes=5, base_filters=64, stem_ks=7, block_ks=3, drop_rate=0.0):
        super().__init__()
        
        self.in_channels = base_filters
        
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=stem_ks, stride=2, 
                     padding=stem_ks//2, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(base_filters, 3, stride=1, kernel_size=block_ks, drop_rate=drop_rate)
        self.layer2 = self._make_layer(base_filters*2, 4, stride=2, kernel_size=block_ks, drop_rate=drop_rate)
        self.layer3 = self._make_layer(base_filters*4, 23, stride=2, kernel_size=block_ks, drop_rate=drop_rate)
        self.layer4 = self._make_layer(base_filters*8, 3, stride=2, kernel_size=block_ks, drop_rate=drop_rate)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.head_drop = nn.Dropout(0.05)
        self.fc = nn.Linear(base_filters*8*2, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride, kernel_size, drop_rate):
        layers = []
        layers.append(BasicBlock1d(self.in_channels, out_channels, stride, kernel_size, drop_rate))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1d(self.in_channels, out_channels, kernel_size=kernel_size, drop_rate=drop_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        
        x = x.flatten(1)
        x = self.head_drop(x)
        x = self.fc(x)
        
        return x

# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

def load_model_from_config(config, num_classes):
    """Load model architecture based on config"""
    mode = config['mode']
    model_type = config['model']
    adapter_strategy = config.get('adapter', 'learned')
    
    if model_type == 'DualStream':
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif model_type == 'CWT2DCNN':
        num_ch = 24 if mode == 'fusion' else 12
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif model_type == 'ViTECG':
        model = ViTECG(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
    elif model_type == 'SwinTransformerECG':
        model = SwinTransformerECG(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
    elif model_type == 'SwinTransformerEarlyFusion':
        model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=False)
    elif model_type == 'SwinTransformerLateFusion':
        model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
    elif model_type == 'EfficientNetECG':
        model = EfficientNetECG(num_classes=num_classes, pretrained=False, adapter_strategy=adapter_strategy)
    elif model_type == 'ViTFusionECG':
        model = ViTFusionECG(num_classes=num_classes, pretrained=False)
    elif model_type == 'EfficientNetFusionECG':
        model = EfficientNetFusionECG(num_classes=num_classes, pretrained=False)
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
            title="ResNet1D-Baseline - Confusion Matrix"
        )
        
        plot_confusion_matrix_multiclass(
            y_true, y_pred, metadata['classes'],
            save_path=os.path.join(ENSEMBLE_PATH, "confusion_multiclass_ResNet1D-Baseline.png"),
            title="ResNet1D-Baseline - Multi-Class Confusion Matrix"
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
        
        plot_combined_confusion(
            y_true, y_pred, metadata['classes'],
            save_path=os.path.join(ENSEMBLE_PATH, f"confusion_combined_{model_name}.png"),
            title=f"{model_name} - Confusion Matrix"
        )
        
        plot_confusion_matrix_multiclass(
            y_true, y_pred, metadata['classes'],
            save_path=os.path.join(ENSEMBLE_PATH, f"confusion_multiclass_{model_name}.png"),
            title=f"{model_name} - Multi-Class Confusion Matrix"
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
    
    # Load metadata
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset info:")
    print(f"  Classes: {metadata['num_classes']} - {metadata['classes']}")
    print(f"  Test samples: {metadata['test_size']}")
    
    # Load test data
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    # Load raw test signals for ResNet1D baseline
    print("\nLoading raw test signals for ResNet1D baseline...")
    X_test_raw = np.load(os.path.join(PROCESSED_PATH, 'X_test_scaled.npy'))
    
    print(f"  Raw signals shape: {X_test_raw.shape}")
    
    # Evaluate all models (CWT + ResNet1D baseline)
    model_results, y_true = evaluate_all_models(metadata, X_test_raw, y_test)
    
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


if __name__ == '__main__':
    main()