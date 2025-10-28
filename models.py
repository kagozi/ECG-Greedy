import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ChannelAdapter(nn.Module):
    """
    Adapts 12-channel ECG input to 3-channel RGB format for pretrained models.
    Three strategies:
    1. 'learned': Learnable 1x1 conv (12→3 channels)
    2. 'pca': Average groups of 4 leads into 3 channels
    3. 'select': Select 3 representative leads
    """
    
    def __init__(self, strategy='learned'):
        super().__init__()
        self.strategy = strategy
        
        if strategy == 'learned':
            # Learnable projection from 12 to 3 channels
            self.adapter = nn.Conv2d(12, 3, kernel_size=1, bias=False)
        elif strategy == 'pca':
            # Fixed grouping: average 4 leads → 1 channel
            # Lead groups: [I,II,III,aVR] → R, [aVL,aVF,V1,V2] → G, [V3,V4,V5,V6] → B
            pass
        elif strategy == 'select':
            # Select 3 key leads: II (limb), V2 (septal), V5 (lateral)
            self.selected_leads = [1, 7, 10]  # indices for leads II, V2, V5
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def forward(self, x):
        # x: (B, 12, H, W)
        if self.strategy == 'learned':
            return self.adapter(x)  # (B, 3, H, W)
        
        elif self.strategy == 'pca':
            # Average groups of 4 consecutive leads
            r = x[:, 0:4, :, :].mean(dim=1, keepdim=True)
            g = x[:, 4:8, :, :].mean(dim=1, keepdim=True)
            b = x[:, 8:12, :, :].mean(dim=1, keepdim=True)
            return torch.cat([r, g, b], dim=1)  # (B, 3, H, W)
        
        elif self.strategy == 'select':
            # Select specific leads
            return x[:, self.selected_leads, :, :]  # (B, 3, H, W)
        
        
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
    """
    2D CNN for CWT representations
    Treats 12 ECG leads as input channels
    """
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Pooling (combine avg and max)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
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
        # x: (B, channels, H, W)
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
    """
    Dual-stream CNN for scalogram + phasogram fusion
    Two parallel branches that share no weights
    """
    
    def __init__(self, num_classes=5, num_channels=12):
        super().__init__()
        
        # Two independent branches
        self.scalogram_branch = CWT2DCNN(num_classes, num_channels)
        self.phasogram_branch = CWT2DCNN(num_classes, num_channels)
        
        # Remove final FC layers from branches
        self.scalogram_branch.fc = nn.Identity()
        self.phasogram_branch.fc = nn.Identity()
        
        # Fusion head
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),  # *2 for concat pooling, *2 for two branches
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


class SwinTransformerECG(nn.Module):
    """
    Swin Transformer for ECG scalogram or phasogram classification.
    Works with 12-channel inputs (scalogram OR phasogram).
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, 
                 model_name='swin_large_patch4_window7_224', adapter_strategy='learned'):
        super().__init__()
        
        # Channel adapter: 12 → 3
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        
        # Load pretrained Swin Transformer from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            in_chans=3      # Standard 3-channel input after adapter
        )
        
        # Get number of features
        num_features = self.backbone.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  SwinTransformerECG: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, x):
        # x: (B, 12, H, W) → (B, 3, H, W)
        x = self.adapter(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class SwinTransformerEarlyFusion(nn.Module):
    """
    Swin Transformer with early fusion for scalogram + phasogram.
    Concatenates 12-channel scalogram + 12-channel phasogram = 24 channels
    Then adapts to 3 channels for pretrained backbone.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_large_patch4_window7_224'):
        super().__init__()
        
        # Adapter for 24 channels (12 scalo + 12 phaso) → 3 channels
        self.adapter = nn.Conv2d(24, 3, kernel_size=1, bias=False)
        
        # Load pretrained Swin Transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  SwinTransformerEarlyFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # x: (B, 24, H, W) from fusion dataset mode
        x = self.adapter(x)  # (B, 3, H, W)
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class SwinTransformerLateFusion(nn.Module):
    """
    Swin Transformer with late fusion.
    Two separate backbones (one for scalogram, one for phasogram)
    with feature-level fusion.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_large_patch4_window7_224', adapter_strategy='learned'):
        super().__init__()
        
        # Two separate channel adapters
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        # Two separate backbones
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
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  SwinTransformerLateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        # scalogram: (B, 12, H, W)
        # phasogram: (B, 12, H, W)
        
        # Adapt channels
        scalo_3ch = self.adapter_scalo(scalogram)  # (B, 3, H, W)
        phaso_3ch = self.adapter_phaso(phasogram)  # (B, 3, H, W)
        
        # Extract features separately
        features_scalo = self.backbone_scalogram(scalo_3ch)
        features_phaso = self.backbone_phasogram(phaso_3ch)
        
        # Concatenate and fuse
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output
    

class EfficientNetECG(nn.Module):
    """
    EfficientNet for ECG - lighter and faster than transformers.
    Good for comparison and faster training.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='efficientnet_b2', adapter_strategy='learned'):
        super().__init__()
        
        # Channel adapter: 12 → 3
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        # Custom classification head
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
        # x: (B, 12, H, W) → (B, 3, H, W)
        x = self.adapter(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    

class ViTECG(nn.Module):
    """Vision Transformer for ECG classification with 12-channel input"""
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, adapter_strategy='learned'):
        super().__init__()
        
        # Channel adapter: 12 → 3
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        
        # Load pretrained ViT-B/16
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.backbone = vit_b_16(weights=weights)
        else:
            self.backbone = vit_b_16(weights=None)
        
        # Get number of features
        num_features = self.backbone.heads.head.in_features
        
        # Replace head with custom classifier
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
        # x: (B, 12, H, W) → (B, 3, H, W)
        x = self.adapter(x)
        return self.backbone(x)
    
    
# ============================================================================
# ViT LATE FUSION
# ============================================================================

class ViTLateFusion(nn.Module):
    """
    Vision Transformer with late fusion.
    Two separate backbones (one for scalogram, one for phasogram)
    with feature-level fusion.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, adapter_strategy='learned'):
        super().__init__()
        
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        
        # Two separate channel adapters
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        # Two separate ViT backbones
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            self.backbone_scalogram = vit_b_16(weights=weights)
            self.backbone_phasogram = vit_b_16(weights=weights)
        else:
            self.backbone_scalogram = vit_b_16(weights=None)
            self.backbone_phasogram = vit_b_16(weights=None)
        
        # Get number of features from ViT
        num_features = self.backbone_scalogram.heads.head.in_features
        
        # Remove classification heads (we'll add our own)
        self.backbone_scalogram.heads.head = nn.Identity()
        self.backbone_phasogram.heads.head = nn.Identity()
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ViTLateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        # scalogram: (B, 12, H, W)
        # phasogram: (B, 12, H, W)
        
        # Adapt channels
        scalo_3ch = self.adapter_scalo(scalogram)  # (B, 3, H, W)
        phaso_3ch = self.adapter_phaso(phasogram)  # (B, 3, H, W)
        
        # Extract features separately
        features_scalo = self.backbone_scalogram(scalo_3ch)
        features_phaso = self.backbone_phasogram(phaso_3ch)
        
        # Concatenate and fuse
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output


# ============================================================================
# EFFICIENTNET LATE FUSION
# ============================================================================

class EfficientNetLateFusion(nn.Module):
    """
    EfficientNet with late fusion.
    Two separate backbones (one for scalogram, one for phasogram)
    with feature-level fusion.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='efficientnet_b2', adapter_strategy='learned'):
        super().__init__()
        
        # Two separate channel adapters
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        # Two separate EfficientNet backbones
        self.backbone_scalogram = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=3
        )
        
        self.backbone_phasogram = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=3
        )
        
        # Get number of features
        num_features = self.backbone_scalogram.num_features
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  EfficientNetLateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        # scalogram: (B, 12, H, W)
        # phasogram: (B, 12, H, W)
        
        # Adapt channels
        scalo_3ch = self.adapter_scalo(scalogram)  # (B, 3, H, W)
        phaso_3ch = self.adapter_phaso(phasogram)  # (B, 3, H, W)
        
        # Extract features separately
        features_scalo = self.backbone_scalogram(scalo_3ch)
        features_phaso = self.backbone_phasogram(phaso_3ch)
        
        # Concatenate and fuse
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output


# SE Attention Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        scale = self.global_avg_pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale

# ============================================================================
# SE BLOCK (needed for hybrid models)
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
# 1. HYBRID SWIN TRANSFORMER ECG (Single Modality)
# ============================================================================

class HybridSwinTransformerECG(nn.Module):
    """
    Hybrid CNN-Swin Transformer for ECG classification.
    CNN stem extracts local features before Swin processes global patterns.
    Works with 12-channel inputs (scalogram OR phasogram).
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, 
                 model_name='swin_base_patch4_window7_224', adapter_strategy='learned'):
        super().__init__()
        
        # Channel adapter: 12 → 3
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        
        # CNN stem for local feature extraction (maintains spatial dimensions)
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
        
        # Load pretrained Swin Transformer
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=3
        )
        
        num_features = self.swin.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  HybridSwinTransformerECG: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, x):
        # x: (B, 12, H, W) → (B, 3, H, W)
        x = self.adapter(x)
        
        # CNN stem extracts local features
        x = self.conv_stem(x)  # Still (B, 3, 224, 224)
        
        # Swin processes global patterns
        features = self.swin(x)
        
        # Classification
        output = self.classifier(features)
        return output


# ============================================================================
# 2. HYBRID SWIN TRANSFORMER EARLY FUSION (Scalogram + Phasogram)
# ============================================================================

class HybridSwinTransformerEarlyFusion(nn.Module):
    """
    Hybrid CNN-Swin Transformer with early fusion.
    Concatenates 12-channel scalogram + 12-channel phasogram = 24 channels,
    then CNN stem processes combined features before Swin.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224'):
        super().__init__()
        
        # Adapter for 24 channels (12 scalo + 12 phaso) → 3 channels
        self.adapter = nn.Conv2d(24, 3, kernel_size=1, bias=False)
        
        # CNN stem for local feature extraction
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
        
        # Load pretrained Swin Transformer
        self.swin = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.swin.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  HybridSwinTransformerEarlyFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # x: (B, 24, H, W) from fusion dataset mode
        x = self.adapter(x)  # (B, 3, H, W)
        
        # CNN stem extracts local features from fused representation
        x = self.conv_stem(x)  # Still (B, 3, 224, 224)
        
        # Swin processes global patterns
        features = self.swin(x)
        
        # Classification
        output = self.classifier(features)
        return output


# ============================================================================
# 3. HYBRID SWIN TRANSFORMER LATE FUSION (Dual Stream)
# ============================================================================

class HybridSwinTransformerLateFusion(nn.Module):
    """
    Hybrid CNN-Swin Transformer with late fusion.
    Two separate hybrid branches (CNN stem + Swin) for scalogram and phasogram,
    with feature-level fusion at the end.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='swin_base_patch4_window7_224', adapter_strategy='learned'):
        super().__init__()
        
        # Two separate channel adapters
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        # CNN stem for scalogram branch
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
        
        # CNN stem for phasogram branch
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
        
        # Two separate Swin Transformer backbones
        self.swin_scalogram = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        self.swin_phasogram = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.swin_scalogram.num_features
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  HybridSwinTransformerLateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        # scalogram: (B, 12, H, W)
        # phasogram: (B, 12, H, W)
        
        # Adapt channels
        scalo_3ch = self.adapter_scalo(scalogram)  # (B, 3, H, W)
        phaso_3ch = self.adapter_phaso(phasogram)  # (B, 3, H, W)
        
        # Process through CNN stems
        scalo_3ch = self.conv_stem_scalo(scalo_3ch)  # (B, 3, 224, 224)
        phaso_3ch = self.conv_stem_phaso(phaso_3ch)  # (B, 3, 224, 224)
        
        # Extract features with Swin Transformers
        features_scalo = self.swin_scalogram(scalo_3ch)
        features_phaso = self.swin_phasogram(phaso_3ch)
        
        # Concatenate and fuse
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        
        # Classification
        output = self.classifier(fused)
        return output
    
# ============================================================================
# EFFICIENTNET - EARLY FUSION
# ============================================================================

class EfficientNetEarlyFusion(nn.Module):
    """
    EfficientNet with early fusion for scalogram + phasogram.
    Concatenates 12-channel scalogram + 12-channel phasogram = 24 channels
    Then adapts to 3 channels for pretrained backbone.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='efficientnet_b2'):
        super().__init__()
        
        # Adapter for 24 channels (12 scalo + 12 phaso) → 3 channels
        self.adapter = nn.Conv2d(24, 3, kernel_size=1, bias=False)
        
        # Load pretrained EfficientNet
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  EfficientNetEarlyFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # x: (B, 24, H, W) from fusion dataset mode
        x = self.adapter(x)  # (B, 3, H, W)
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# EFFICIENTNET - LATE FUSION (Dual Stream)
# ============================================================================

class EfficientNetLateFusion(nn.Module):
    """
    EfficientNet with late fusion.
    Two separate backbones (one for scalogram, one for phasogram)
    with feature-level fusion.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True,
                 model_name='efficientnet_b2', adapter_strategy='learned'):
        super().__init__()
        
        # Two separate channel adapters
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        # Two separate EfficientNet backbones
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
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  EfficientNetLateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        # scalogram: (B, 12, H, W)
        # phasogram: (B, 12, H, W)
        
        # Adapt channels
        scalo_3ch = self.adapter_scalo(scalogram)  # (B, 3, H, W)
        phaso_3ch = self.adapter_phaso(phasogram)  # (B, 3, H, W)
        
        # Extract features separately
        features_scalo = self.backbone_scalogram(scalo_3ch)
        features_phaso = self.backbone_phasogram(phaso_3ch)
        
        # Concatenate and fuse
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output


# ============================================================================
# RESNET50 - SINGLE MODALITY
# ============================================================================

class ResNet50ECG(nn.Module):
    """
    ResNet50 for ECG - robust CNN baseline.
    Works with 12-channel inputs (scalogram OR phasogram).
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, adapter_strategy='learned'):
        super().__init__()
        
        # Channel adapter: 12 → 3
        self.adapter = ChannelAdapter(strategy=adapter_strategy)
        
        # Load pretrained ResNet50
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=3
        )
        
        num_features = self.backbone.num_features  # 2048 for ResNet50
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ResNet50ECG: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, x):
        # x: (B, 12, H, W) → (B, 3, H, W)
        x = self.adapter(x)
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# RESNET50 - EARLY FUSION
# ============================================================================

class ResNet50EarlyFusion(nn.Module):
    """
    ResNet50 with early fusion for scalogram + phasogram.
    Concatenates 12-channel scalogram + 12-channel phasogram = 24 channels
    Then adapts to 3 channels for pretrained backbone.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True):
        super().__init__()
        
        # Adapter for 24 channels (12 scalo + 12 phaso) → 3 channels
        self.adapter = nn.Conv2d(24, 3, kernel_size=1, bias=False)
        
        # Load pretrained ResNet50
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ResNet50EarlyFusion: {n_params/1e6:.1f}M parameters")
    
    def forward(self, x):
        # x: (B, 24, H, W) from fusion dataset mode
        x = self.adapter(x)  # (B, 3, H, W)
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# ============================================================================
# RESNET50 - LATE FUSION (Dual Stream)
# ============================================================================

class ResNet50LateFusion(nn.Module):
    """
    ResNet50 with late fusion.
    Two separate backbones (one for scalogram, one for phasogram)
    with feature-level fusion.
    """
    
    def __init__(self, num_classes=5, dropout=0.3, pretrained=True, adapter_strategy='learned'):
        super().__init__()
        
        # Two separate channel adapters
        self.adapter_scalo = ChannelAdapter(strategy=adapter_strategy)
        self.adapter_phaso = ChannelAdapter(strategy=adapter_strategy)
        
        # Two separate ResNet50 backbones
        self.backbone_scalogram = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        self.backbone_phasogram = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        
        num_features = self.backbone_scalogram.num_features  # 2048
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ResNet50LateFusion: {n_params/1e6:.1f}M parameters (adapter={adapter_strategy})")
    
    def forward(self, scalogram, phasogram):
        # scalogram: (B, 12, H, W)
        # phasogram: (B, 12, H, W)
        
        # Adapt channels
        scalo_3ch = self.adapter_scalo(scalogram)  # (B, 3, H, W)
        phaso_3ch = self.adapter_phaso(phasogram)  # (B, 3, H, W)
        
        # Extract features separately
        features_scalo = self.backbone_scalogram(scalo_3ch)
        features_phaso = self.backbone_phasogram(phaso_3ch)
        
        # Concatenate and fuse
        combined_features = torch.cat([features_scalo, features_phaso], dim=1)
        fused = self.fusion(combined_features)
        output = self.classifier(fused)
        
        return output
