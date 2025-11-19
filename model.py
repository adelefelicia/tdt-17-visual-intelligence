import torch.nn as nn
from monai.networks.nets import DenseNet121


class BreastMRIClassifier(nn.Module):
    """
    DenseNet-based classifier for breast MRI lesion classification.
    
    Outputs logits for lesion classes (not softmaxed).
    """
    
    def __init__(self, in_channels, num_classes, dropout_prob, pretrained = False):
        """
        Args:
            in_channels: Number of MRI sequences
            num_classes: Number of lesion classes
            dropout_prob: Dropout probability before final layer
            pretrained: Whether to use pretrained weights (if available)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        self.backbone = DenseNet121(
            spatial_dims=3, # 3D volumes
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=dropout_prob,
            pretrained=pretrained
        )
            
    def forward(self, x):
        logits = self.backbone(x)
        return logits
