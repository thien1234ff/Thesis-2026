import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaMSELoss(nn.Module):
    """
    Beta-Weighted Mean Squared Error (Beta-MSE) Loss for embedding alignment
    between Teacher (full precision) and Student (quantized) models.
    
    Attributes:
        beta (float): Gradient amplification coefficient to coordinación the student's 
                     embedding updates. Typical values are between 200 and 500.
    """
    def __init__(self, beta=300.0):
        super(BetaMSELoss, self).__init__()
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, feat_student, feat_teacher):
        """
        Calculates the Beta-MSE loss after normalizing the student and teacher embeddings.
        
        Args:
            feat_student (torch.Tensor): Unnormalized student embeddings (Batch, Dim).
            feat_teacher (torch.Tensor): Unnormalized teacher embeddings (Batch, Dim).
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        # Ensure L2 normalization on the unit hypersphere
        norm_s = F.normalize(feat_student, p=2, dim=1)
        norm_t = F.normalize(feat_teacher, p=2, dim=1)
        
        # Raw MSE distance multiplied by beta for gradient coordination
        return self.beta * self.mse(norm_s, norm_t)
