import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from src.losses import BetaMSELoss
from src.quantizer import freeze_model, unfreeze_model

class QATTrainingEngine:
    """
    Proposed 3-Phase QAT training engine for Face Recognition Quantization.
    Combines warm-up, observer freezing, cool-down fine-tuning, and Beta-MSE alignment.
    """
    def __init__(self, student, teacher, train_loader, optimizer,
                 base_lr=1e-3, beta=300.0, epochs=30, device="cuda"):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.beta = beta
        self.epochs = epochs
        self.device = device
        
        self.criterion = BetaMSELoss(beta=self.beta)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Determine milestones based on epoch scale
        self.warmup_epochs = max(1, int(0.16 * epochs))       # Default: 5 for 30 epochs
        self.cooldown_start = max(2, int(0.83 * epochs))     # Default: 25 for 30 epochs
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._lr_schedule)

    def _lr_schedule(self, epoch):
        """3-phase learning rate multiplier scheduler."""
        if epoch < self.warmup_epochs:
            # Linear warm-up from 1e-6 to base_lr
            start_mult = 1e-6 / self.base_lr
            end_mult = 1.0
            divisor = max(1.0, float(self.warmup_epochs - 1))
            return start_mult + (end_mult - start_mult) * (epoch / divisor)
            
        elif epoch < self.cooldown_start:
            # Main training at base_lr
            return 1.0
            
        else:
            # Cool-down fine-tuning with cosine annealing to 1e-5
            curr = epoch - self.cooldown_start
            total = self.epochs - self.cooldown_start
            cos_out = 0.5 * (1.0 + math.cos(math.pi * curr / max(1.0, total)))
            min_lr_mult = 1e-5 / self.base_lr
            return min_lr_mult + (1.0 - min_lr_mult) * cos_out

    def freeze_student_observers(self):
        """Freezes all activation quantization scale observers in the student model."""
        freeze_model(self.student)

    def train_epoch(self, epoch):
        """Executes one epoch of training."""
        self.student.train()
        
        # Check training phase and freeze observers if entering Phase 2 or 3
        if epoch >= self.warmup_epochs:
            self.freeze_student_observers()
            phase_name = "Main/Fine-Tuning (Observers Frozen)"
        else:
            unfreeze_model(self.student)
            phase_name = "Warm-up (Observers active)"
            
        total_loss = 0.0
        total_acc = 0.0
        total_batches = len(self.train_loader)
        
        for batch_idx, (img, _) in enumerate(self.train_loader):
            img = img.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                feat_t = self.teacher(img)
                
            feat_s = self.student(img)
            
            # Compute gradient-coordinated alignment loss
            loss = self.criterion(feat_s, feat_t)
            
            # Calculate alignment accuracy (sim > 0.5)
            with torch.no_grad():
                norm_s = F.normalize(feat_s, dim=1)
                norm_t = F.normalize(feat_t, dim=1)
                cos_sim = (norm_s * norm_t).sum(dim=1)
                acc = (cos_sim > 0.5).float().mean().item()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch+1}/{self.epochs}] | Batch [{batch_idx+1}/{total_batches}] | "
                      f"Phase: {phase_name} | Loss: {loss.item():.4f} | Alignment Acc: {acc*100:.2f}%")
                      
        self.scheduler.step()
        
        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        return avg_loss, avg_acc

    def train(self, save_dir="weights/w-3p", model_prefix="iresnet18"):
        """Full training sequence loop."""
        os.makedirs(save_dir, exist_ok=True)
        epoch_losses = []
        epoch_accs = []
        
        print(f"Starting QAT Training (Epochs: {self.epochs}, Base LR: {self.base_lr}, Beta: {self.beta})")
        print(f"Warm-up: {self.warmup_epochs} epochs | Main: {self.cooldown_start - self.warmup_epochs} epochs | Cool-down: {self.epochs - self.cooldown_start} epochs")
        
        for epoch in range(self.epochs):
            loss, acc = self.train_epoch(epoch)
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            
            print(f"==> Epoch [{epoch+1}/{self.epochs}] Summary | Avg Loss: {loss:.4f} | Avg Acc: {acc*100:.2f}% | LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(save_dir, f"{model_prefix}_q6_epoch{epoch}.pth")
            torch.save(self.student.state_dict(), checkpoint_path)
            
        return epoch_losses, epoch_accs
