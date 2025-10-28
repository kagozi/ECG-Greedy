## New focal_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# IMPROVED FOCAL LOSS (Unified for Multi-Label)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Unified Focal Loss for binary, multi-class, and multi-label classification.
    Optimized for multi-label ECG classification.
    """
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='multi-label', num_classes=None):
        """
        :param gamma: Focusing parameter (default: 2)
        :param alpha: Balancing factor for class weights (scalar or tensor)
        :param reduction: 'none' | 'mean' | 'sum'
        :param task_type: 'binary' | 'multi-class' | 'multi-label'
        :param num_classes: Number of classes (for multi-class only)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        :param inputs: Predictions (logits) - Shape: (batch_size, num_classes)
        :param targets: Ground truth labels - Shape: (batch_size, num_classes) for multi-label
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(f"Unsupported task_type '{self.task_type}'")

    def binary_focal_loss(self, inputs, targets):
        """Focal loss for binary classification"""
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multi-class classification"""
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        ce_loss = -targets_one_hot * torch.log(probs + 1e-7)
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """Focal loss for multi-label classification (ECG use case)"""
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# DISTRIBUTION-AWARE FOCAL LOSS (With Class Weights)
# ============================================================================

class DistributionAwareFocalLoss(nn.Module):
    """
    Focal Loss with per-class weighting for handling imbalanced datasets.
    Best for ECG multi-label classification with varying class frequencies.
    """
    
    def __init__(self, class_weights=None, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        :param inputs: Logits (batch_size, num_classes)
        :param targets: Binary labels (batch_size, num_classes)
        """
        probs = torch.sigmoid(inputs)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply all weights
        loss = alpha_t * focal_weight * bce_loss
        
        # Apply class weights if provided
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(inputs.device)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    '''
    Example usage:
    import torch
    from focal_loss import FocalLoss

    num_classes = 5
    criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-label')
    inputs = torch.randn(16, num_classes)  # Logits from the model
    targets = torch.randint(0, 2, (16, num_classes)).float()  # Ground truth labels

    loss = criterion(inputs, targets)
    print(f'Multi-label Focal Loss: {loss.item()}')
    '''