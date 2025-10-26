# ============================================================================
# STEP 3: Train CNN Models on CWT Representations
# ============================================================================
# Tests multiple model architectures with Focal Loss support
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import fbeta_score, roc_auc_score, f1_score, roc_curve, confusion_matrix
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
import seaborn as sns



# ============================================================================
# CONFIGURATION
# ============================================================================

PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
PROCESSED_PATH = '../santosh_lab/shared/KagoziA/wavelets/xresnet_baseline/'
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

print("="*80)
print("STEP 3: TRAIN CNN MODELS ON CWT REPRESENTATIONS")
print("="*80)
print(f"Device: {DEVICE}")

# ============================================================================
# THRESHOLD FUNCTIONS
# ============================================================================

def find_optimal_thresholds(y_true, y_scores):
    """Find optimal thresholds per class using ROC curve"""
    thresholds = []
    for i in range(y_true.shape[1]):
        fpr, tpr, threshold = roc_curve(y_true[:, i], y_scores[:, i])
        optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
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


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label classification
    Focuses on hard examples by down-weighting easy ones
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: (B, num_classes) logits
        # targets: (B, num_classes) binary labels
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# DATASET CLASS WITH AUGMENTATION
# ============================================================================

class CWTDataset(Dataset):
    """Memory-efficient dataset that loads CWT data on-the-fly with optional augmentation"""
    
    def __init__(self, scalo_path, phaso_path, labels, mode='scalogram', augment=False):
        self.scalograms = np.load(scalo_path, mmap_mode='r', allow_pickle=True)
        self.phasograms = np.load(phaso_path, mmap_mode='r', allow_pickle=True)
        self.labels = torch.FloatTensor(labels)
        self.mode = mode
        self.augment = augment
        
        print(f"  Dataset loaded: {len(self.labels)} samples, mode={mode}, augment={augment}")
        print(f"  Scalograms shape: {self.scalograms.shape}")
        print(f"  Phasograms shape: {self.phasograms.shape}")
    
    def __len__(self):
        return len(self.labels)
    
    def _augment_image(self, img):
        """Apply random augmentations to CWT image"""
        # img: (C, H, W) tensor
        
        # Random horizontal flip (50% chance)
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])
        
        # Random vertical flip (30% chance) - less aggressive
        if torch.rand(1).item() > 0.7:
            img = torch.flip(img, dims=[1])
        
        # Random brightness adjustment (±10%)
        if torch.rand(1).item() > 0.5:
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            img = torch.clamp(img * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if torch.rand(1).item() > 0.5:
            mean = img.mean()
            contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            img = torch.clamp((img - mean) * contrast_factor + mean, 0, 1)
        
        # Add small Gaussian noise (5% chance)
        if torch.rand(1).item() > 0.95:
            noise = torch.randn_like(img) * 0.01
            img = torch.clamp(img + noise, 0, 1)
        
        return img
    
    def __getitem__(self, idx):
        scalo = torch.FloatTensor(np.array(self.scalograms[idx], copy=True))
        phaso = torch.FloatTensor(np.array(self.phasograms[idx], copy=True))
        label = self.labels[idx]
        
        # Apply augmentation if enabled (only for training)
        if self.augment:
            scalo = self._augment_image(scalo)
            phaso = self._augment_image(phaso)
        
        if self.mode == 'scalogram':
            return scalo, label
        elif self.mode == 'phasogram':
            return phaso, label
        elif self.mode == 'both':
            return (scalo, phaso), label
        elif self.mode == 'fusion':
            fused = torch.cat([scalo, phaso], dim=0)
            return fused, label
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ============================================================================
# SE BLOCK FOR HYBRID MODELS
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ============================================================================
# CHANNEL ADAPTER
# ============================================================================

class ChannelAdapter(nn.Module):
    """Adapts 12-channel ECG input to 3-channel RGB format for pretrained models"""
    
    def __init__(self, strategy='learned'):
        super().__init__()
        self.strategy = strategy
        
        if strategy == 'learned':
            self.adapter = nn.Conv2d(12, 3, kernel_size=1, bias=False)
        elif strategy == 'select':
            self.selected_leads = [1, 7, 10]  # leads II, V2, V5
    
    def forward(self, x):
        if self.strategy == 'learned':
            return self.adapter(x)
        elif self.strategy == 'pca':
            r = x[:, 0:4, :, :].mean(dim=1, keepdim=True)
            g = x[:, 4:8, :, :].mean(dim=1, keepdim=True)
            b = x[:, 8:12, :, :].mean(dim=1, keepdim=True)
            return torch.cat([r, g, b], dim=1)
        elif self.strategy == 'select':
            return x[:, self.selected_leads, :, :]

class Normalize(nn.Module):
    def __init__(self, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor(std).view(1,3,1,1))
    def forward(self, x):  # x: (B,3,H,W)
        return (x - self.mean) / self.std

# ============================================================================
# PRETRAINED MODELS
# ============================================================================

class ViTECG(nn.Module):
    """Vision Transformer for ECG classification with 12-channel input"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, adapter_strategy='learned'):
        super().__init__()
        
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        self.norm = Normalize()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)
        
        num_features = self.backbone.heads.head.in_features
        
        self.backbone.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ViTECG: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, x):
        x = self.adapter(x)
        # x = self.norm(x)
        return self.backbone(x)


class SwinTransformerECG(nn.Module):
    """Swin Transformer for ECG classification with hybrid CNN stem"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, 
                 model_name='swin_base_patch4_window7_224', adapter_strategy='learned',
                 use_hybrid=True):
        super().__init__()
        
        self.use_hybrid = use_hybrid
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        self.norm = Normalize()
        
        if use_hybrid:
            # Hybrid CNN stem: enhances local feature extraction before Swin
            self.conv_stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SEBlock(128),
                nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        hybrid_str = "Hybrid-" if use_hybrid else ""
        print(f"  {hybrid_str}SwinTransformerECG: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, x):
        x = self.adapter(x)  # 12 channels → 3 channels

        
        if self.use_hybrid:
            x = self.conv_stem(x)  # CNN feature extraction (maintains 224x224)
 
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class SwinTransformerEarlyFusion(nn.Module):
    """Swin Transformer with early fusion for scalogram + phasogram with hybrid CNN stem"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224', use_hybrid=False):
        super().__init__()
        
        self.use_hybrid = use_hybrid
        self.adapter = nn.Conv2d(24, 3, kernel_size=1, bias=False)
        
        if use_hybrid:
            # Hybrid CNN stem for fusion
            self.conv_stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SEBlock(128),
                nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        hybrid_str = "Hybrid-" if use_hybrid else ""
        print(f"  {hybrid_str}SwinTransformerEarlyFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        x = self.adapter(x)  # 24 channels → 3 channels
        # x = self.norm(x)
        if self.use_hybrid:
            x = self.conv_stem(x)  # CNN feature extraction
        
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class SwinTransformerLateFusion(nn.Module):
    """Swin Transformer with late fusion (dual stream) with hybrid CNN stems"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224', adapter_strategy='learned',
                 use_hybrid=True):
        super().__init__()
        
        self.use_hybrid = use_hybrid
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        if use_hybrid:
            # Separate hybrid CNN stems for scalogram and phasogram
            self.conv_stem_scalo = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SEBlock(128),
                nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
            
            self.conv_stem_phaso = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SEBlock(128),
                nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        
        self.backbone_scalogram = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        self.backbone_phasogram = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone_scalogram.num_features
        
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        hybrid_str = "Hybrid-" if use_hybrid else ""
        print(f"  {hybrid_str}SwinTransformerLateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        scalo_3ch = self.adapter_scalo(scalogram)  # 12 → 3
        phaso_3ch = self.adapter_phaso(phasogram)  # 12 → 3
        
        if self.use_hybrid:
            scalo_3ch = self.conv_stem_scalo(scalo_3ch)  # CNN features
            phaso_3ch = self.conv_stem_phaso(phaso_3ch)  # CNN features
        
        features_scalo = self.backbone_scalogram(scalo_3ch)
        features_phaso = self.backbone_phasogram(phaso_3ch)
        
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output


class EfficientNetECG(nn.Module):
    """EfficientNet for ECG - lighter and faster than transformers"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='efficientnet_b2', adapter_strategy='learned'):
        super().__init__()
        
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  EfficientNetECG: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, x):
        x = self.adapter(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# CNN BASELINE MODELS
# ============================================================================

class ResidualBlock2D(nn.Module):
    """Residual block for 2D CNN"""
    
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class CWT2DCNN(nn.Module):
    """2D CNN for CWT representations"""
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  CWT2DCNN: {n_params/1e6:.1f}M parameters")
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_ch, out_ch, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        return ResidualBlock2D(in_ch, out_ch, stride, downsample)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        
        return self.fc(x)


class DualStreamCNN(nn.Module):
    """Dual-stream CNN for scalogram + phasogram fusion"""
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        self.scalogram_branch = CWT2DCNN(num_classes, num_channels)
        self.phasogram_branch = CWT2DCNN(num_classes, num_channels)
        
        self.scalogram_branch.fc = nn.Identity()
        self.phasogram_branch.fc = nn.Identity()
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  DualStreamCNN: {n_params/1e6:.1f}M parameters")
    
    def forward(self, scalogram, phasogram):
        feat_scalo = self.scalogram_branch(scalogram)
        feat_phaso = self.phasogram_branch(phasogram)
        
        combined = torch.cat([feat_scalo, feat_phaso], dim=1)
        return self.fusion_fc(combined)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

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

def train_epoch(model, dataloader, criterion, optimizer, device, is_dual=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        if is_dual:
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x1, x2)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
        
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * y.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device, is_dual=False):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        if is_dual:
            (x1, x2), y = batch
            x1, x2 = x1.to(device), x2.to(device)
            out = model(x1, x2)
        else:
            x, y = batch
            x = x.to(device)
            out = model(x)
        
        loss = criterion(out, y.to(device))
        running_loss += loss.item() * y.size(0)
        
        probs = torch.sigmoid(out).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(y.numpy())
    
    return running_loss / len(dataloader.dataset), np.vstack(all_preds), np.vstack(all_labels)


def compute_metrics(y_true, y_pred, y_scores):
    """Compute evaluation metrics"""
    try:
        macro_auc = roc_auc_score(y_true, y_scores, average='macro')
    except:
        macro_auc = 0.0
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f_beta = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
    
    return {
        'macro_auc': macro_auc,
        'f1_macro': f1_macro,
        'f_beta_macro': f_beta
    }
    
def compute_pos_weight(y_train):
    # y_train: (N,C) binary numpy array
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    w = (neg / np.clip(pos, 1, None))
    return torch.tensor(w, dtype=torch.float32)



# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_model(config, metadata, device):
    """Train a single model configuration"""
    
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")
    
    # Load labels
    y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train.npy'))
    y_val = np.load(os.path.join(PROCESSED_PATH, 'y_val.npy'))
    y_test = np.load(os.path.join(PROCESSED_PATH, 'y_test.npy'))
    
    # Create datasets
    mode = config['mode']
    is_dual = (config['model'] in ['DualStream', 'SwinTransformerLateFusion'])
    
    print(f"\nCreating datasets (mode={mode})...")
    train_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'train_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'train_phasograms.npy'),
        y_train, mode=mode, augment=False  # Enable augmentation for training
    )
    val_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'val_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'val_phasograms.npy'),
        y_val, mode=mode, augment=False  # No augmentation for validation
    )
    test_dataset = CWTDataset(
        os.path.join(PROCESSED_PATH, 'test_scalograms.npy'),
        os.path.join(PROCESSED_PATH, 'test_phasograms.npy'),
        y_test, mode=mode, augment=False  # No augmentation for testing
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Create model with adapter strategy support
    print(f"\nCreating model...")
    num_classes = metadata['num_classes']
    adapter_strategy = config.get('adapter', 'learned')  # Get adapter strategy from config
    
    if config['model'] == 'DualStreamCNN':
        model = DualStreamCNN(num_classes=num_classes, num_channels=12)
    elif config['model'] == 'CWT2DCNN':
        num_ch = 24 if mode == 'fusion' else 12
        model = CWT2DCNN(num_classes=num_classes, num_channels=num_ch)
    elif config['model'] == 'ViTECG':
        model = ViTECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'SwinTransformerECG':
        model = SwinTransformerECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'SwinTransformerEarlyFusion':
        model = SwinTransformerEarlyFusion(num_classes=num_classes, pretrained=True)
    elif config['model'] == 'SwinTransformerLateFusion':
        model = SwinTransformerLateFusion(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    elif config['model'] == 'EfficientNetECG':
        model = EfficientNetECG(num_classes=num_classes, pretrained=True, adapter_strategy=adapter_strategy)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    model = model.to(device)
    
    # Choose loss function
    loss_type = config.get('loss', 'bce')
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print(f"Using Focal Loss (alpha=0.25, gamma=2.0)")
    else:
        # criterion = nn.BCEWithLogitsLoss()
        pw = compute_pos_weight(y_train).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"Using BCE Loss")
    
    # Optimizer with different LR for pretrained models
    pretrained_models = ['ViTECG', 'SwinTransformerECG', 'EfficientNetECG', 
                        'SwinTransformerLateFusion', 'SwinTransformerEarlyFusion']
    
    if config['model'] in pretrained_models:
        if 'Swin' in config['model']:
            lr = 3e-5  # Lower LR for finetuning
        else: 
            lr = 1e-4 
        print(f"Using LR={lr} (finetuning)")
    else:
        lr = LR
        print(f"Using LR={lr}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\nTraining for {EPOCHS} epochs...")
    best_val_auc = 0.0
    best_thresholds = None
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, is_dual)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device, is_dual)
        
        # Find optimal thresholds and compute metrics
        thresholds = find_optimal_thresholds(val_labels, val_preds)
        val_pred_binary = apply_thresholds(val_preds, thresholds)
        val_metrics = compute_metrics(val_labels, val_pred_binary, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_metrics['macro_auc'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val AUC: {val_metrics['macro_auc']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
        
        # Save best model and thresholds
        if val_metrics['macro_auc'] > best_val_auc:
            best_val_auc = val_metrics['macro_auc']
            best_thresholds = thresholds.copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'thresholds': best_thresholds,
                'config': config
            }, os.path.join(PROCESSED_PATH, f"best_{config['name']}.pth"))
            print(f"✓ Saved best model (AUC: {best_val_auc:.4f})")
        
        scheduler.step(val_metrics['macro_auc'])
        
        # Early stopping
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print("Learning rate too small, stopping early")
            break
    
    # Test with best model
    print(f"\nTesting {config['name']}...")
    checkpoint = torch.load(os.path.join(PROCESSED_PATH, f"best_{config['name']}.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device, is_dual)
    
    # Apply optimal thresholds
    test_pred_binary = apply_thresholds(test_preds, best_thresholds)
    test_metrics = compute_metrics(test_labels, test_pred_binary, test_preds)
    
    print(f"\nTest Results - {config['name']}:")
    print(f"  AUC:    {test_metrics['macro_auc']:.4f}")
    print(f"  F1:     {test_metrics['f1_macro']:.4f}")
    print(f"  F-beta: {test_metrics['f_beta_macro']:.4f}")
    
    try:
        plot_confusion_matrix_all_classes(
            test_labels, 
            test_pred_binary, 
            metadata['classes'],
            save_path=os.path.join(PROCESSED_PATH, f"confusion_matrix_{config['name']}.png"),
            title=f"Confusion Matrix - {config['name']}"
        )
        print(f"✓ Confusion matrix saved: confusion_matrix_{config['name']}.png")
    except Exception as e:
        print(f"❌ Error generating confusion matrix: {e}")
    
    # Save results
    results = {
        'config': config,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'optimal_thresholds': best_thresholds.tolist(),
        'history': history
    }
    
    with open(os.path.join(PROCESSED_PATH, f"results_{config['name']}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    # Load metadata
    print("\n[1/2] Loading metadata...")
    with open(os.path.join(PROCESSED_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Dataset info:")
    print(f"  Classes: {metadata['num_classes']} - {metadata['classes']}")
    print(f"  Train: {metadata['train_size']} samples")
    print(f"  Val:   {metadata['val_size']} samples")
    print(f"  Test:  {metadata['test_size']} samples")
    
    # Define model configurations to train
    configs = [    
        # {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal', 'loss': 'focal'},
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'fusion', 'model': 'DualStreamCNN', 'name': 'DualStreamCNN-BCE', 'loss': 'bce'},
                # {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal', 'loss': 'focal'},
        
        # Baseline CNN models with both losses
        # Fusion models
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-BCE', 'loss': 'bce'},
        # {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal', 'loss': 'focal'},
        

    
        
        # # Pretrained models - compare BCE vs Focal Loss
        # {'mode': 'scalogram', 'model': 'EfficientNetECG', 'name': 'Scalogram-EfficientNet-BCE', 'loss': 'bce'},
        # {'mode': 'scalogram', 'model': 'EfficientNetECG', 'name': 'Scalogram-EfficientNet-Focal', 'loss': 'focal'},
        
        # {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Scalogram-Swin-BCE', 'loss': 'bce'},
        # {'mode': 'scalogram', 'model': 'SwinTransformerECG', 'name': 'Scalogram-Swin-Focal', 'loss': 'focal'},

    ]
    
    # Train all models
    print("\n[2/2] Training models...")
    all_results = {}
    
    for config in configs:
        try:
            results = train_model(config, metadata, DEVICE)
            all_results[config['name']] = results['test_metrics']
        except Exception as e:
            print(f"\n❌ Error training {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final comparison with adapter strategy analysis
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    print(f"{'Model':<50} | {'Adapter':<8} | {'AUC':<8} | {'F1':<8} | {'F-beta':<8}")
    print("-" * 100)
    
    # Group by model type
    for name, metrics in sorted(all_results.items()):
        adapter = 'N/A'
        if 'Learned' in name:
            adapter = 'Learned'
        elif 'Select' in name:
            adapter = 'Select'
        elif 'PCA' in name:
            adapter = 'PCA'
        
        print(f"{name:<50} | {adapter:<8} | {metrics['macro_auc']:.4f}   | "
              f"{metrics['f1_macro']:.4f}   | {metrics['f_beta_macro']:.4f}")
    
    # Save final results
    with open(os.path.join(PROCESSED_PATH, 'final_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Analyze adapter strategies
    print("\n" + "="*80)
    print("ADAPTER STRATEGY COMPARISON")
    print("="*80)
    
    learned_models = {k: v for k, v in all_results.items() if 'Learned' in k}
    select_models = {k: v for k, v in all_results.items() if 'Select' in k}
    pca_models = {k: v for k, v in all_results.items() if 'PCA' in k}
    
    if learned_models:
        avg_auc_learned = np.mean([v['macro_auc'] for v in learned_models.values()])
        avg_f1_learned = np.mean([v['f1_macro'] for v in learned_models.values()])
        print(f"Learned Adapter - Avg AUC: {avg_auc_learned:.4f}, Avg F1: {avg_f1_learned:.4f}")
    
    if select_models:
        avg_auc_select = np.mean([v['macro_auc'] for v in select_models.values()])
        avg_f1_select = np.mean([v['f1_macro'] for v in select_models.values()])
        print(f"Select Adapter  - Avg AUC: {avg_auc_select:.4f}, Avg F1: {avg_f1_select:.4f}")
    
    if pca_models:
        avg_auc_pca = np.mean([v['macro_auc'] for v in pca_models.values()])
        avg_f1_pca = np.mean([v['f1_macro'] for v in pca_models.values()])
        print(f"PCA Adapter     - Avg AUC: {avg_auc_pca:.4f}, Avg F1: {avg_f1_pca:.4f}")
    
    # Find best model overall
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1]['macro_auc'])
        print(f"Model: {best_model[0]}")
        print(f"  AUC:    {best_model[1]['macro_auc']:.4f}")
        print(f"  F1:     {best_model[1]['f1_macro']:.4f}")
        print(f"  F-beta: {best_model[1]['f_beta_macro']:.4f}")
    
    print("\n" + "="*80)
    print("STEP 3 COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {PROCESSED_PATH}")
    print("\nPipeline finished successfully! 🎉")


if __name__ == '__main__':
    main()