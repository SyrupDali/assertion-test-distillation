# src/distillation.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StrongDistillationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, temperature=2.0):
        super(StrongDistillationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.mse = nn.MSELoss()

    def forward(self, student_hidden, teacher_hidden, student_logits, teacher_logits):
        # Match hidden states (final encoder outputs)
        hidden_loss = self.mse(student_hidden, teacher_hidden)

        # Match logits (softened distributions)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        logit_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        # Total loss
        total_loss = self.alpha * hidden_loss + self.beta * logit_loss
        return total_loss, hidden_loss.item(), logit_loss.item()
