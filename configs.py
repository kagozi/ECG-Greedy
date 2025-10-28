#Define model configurations to train
configs = [
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-BCE', 'loss': 'bce'},
        {'mode': 'scalogram', 'model': 'CWT2DCNN', 'name': 'Scalogram-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-BCE', 'loss': 'bce'},
         {'mode': 'fusion', 'model': 'CWT2DCNN', 'name': 'Fusion-2DCNN-Focal', 'loss': 'focal'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-BCE', 'loss': 'bce'},
        {'mode': 'both', 'model': 'DualStream', 'name': 'DualStream-CNN-Focal', 'loss': 'focal'},
              

        
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-BCE-Learned', 'loss': 'bce', 'adapter': 'learned'},
        {'mode': 'fusion', 'model': 'SwinTransformerEarlyFusion', 'name': 'EarlyFusion-Swin-Focal-Select', 'loss': 'focal', 'adapter': 'select'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-BCE-Learned', 'loss': 'bce', 'adapter': 'learned'},
        {'mode': 'both', 'model': 'SwinTransformerLateFusion', 'name': 'LateFusion-Swin-Focal-Select', 'loss': 'focal', 'adapter': 'select'},
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'name': 'EfficientNetLateFusion-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        {'mode': 'both', 'model': 'ViTLateFusion', 'name': 'ViTLateFusion-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
        
        {'mode': 'scalogram', 'model': 'ViTECG', 'name': 'ViT-ECG-BCE-Learned', 'loss': 'bce', 'adapter': 'learned'},
        {'mode': 'scalogram', 'model': 'EfficientNetFusionECG', 'name': 'EfficientNet-ECG-Focal-Learned', 'loss': 'focal', 'adapter': 'learned'},
         
        # Hybrid Swin variants
        {'mode': 'scalogram', 'model': 'HybridSwinTransformerECG', 'adapter': 'learned', 'name': 'Scalogram-HybridSwin-Learned', 'loss': 'focal_weighted'},
        {'mode': 'fusion', 'model': 'HybridSwinTransformerEarlyFusion', 'name': 'EarlyFusion-HybridSwin', 'loss': 'focal_weighted'},
        

        {'mode': 'phasogram', 'model': 'HybridSwinTransformerECG', 'adapter': 'learned', 'name': 'Scalogram-HybridSwin-Learned', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'HybridSwinTransformerLateFusion', 'adapter': 'learned', 'name': 'LateFusion-HybridSwin-Learned', 'loss': 'focal_weighted'},
        
        
        ## ResNet50 variants
        {'mode': 'scalogram', 'model': 'ResNet50ECG', 'adapter': 'learned', 'name': 'Scalogram-ResNet50-Learned', 'loss': 'focal_weighted'},
        {'mode': 'fusion', 'model': 'ResNet50EarlyFusion', 'adapter': 'learned', 'name': 'EarlyFusion-ResNet50-Learned', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'ResNet50LateFusion', 'adapter': 'learned', 'name': 'LateFusion-ResNet50-Learned', 'loss': 'focal_weighted'},
        {'mode': 'phasogram', 'model': 'ResNet50ECG', 'adapter': 'learned', 'name': 'Phasogram-ResNet50-Learned', 'loss': 'focal_weighted'},
        
        ## EfficientNet variants
        {'mode': 'scalogram', 'model': 'EfficientNetFusionECG', 'adapter': 'learned', 'name': 'Scalogram-EfficientNet-Learned', 'loss': 'focal_weighted'},
        {'mode': 'fusion', 'model': 'EfficientNetEarlyFusion', 'adapter': 'learned', 'name': 'EarlyFusion-EfficientNet-Learned', 'loss': 'focal_weighted'},
        {'mode': 'both', 'model': 'EfficientNetLateFusion', 'adapter': 'learned', 'name': 'LateFusion-EfficientNet-Learned', 'loss': 'focal_weighted'},
        {'mode': 'phasogram', 'model': 'EfficientNetFusionECG', 'adapter': 'learned', 'name': 'Phasogram-EfficientNet-Learned', 'loss': 'focal_weighted'},

]