# src/train.py
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer
from src.models import StudentModel
from src.distillation import StrongDistillationLoss

def train(student_model, teacher_model, dataloader, optimizer, device):
    distillation_loss_fn = StrongDistillationLoss()
    student_model.train()
    teacher_model.eval()  # Freeze teacher

    for batch in tqdm(dataloader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            teacher_hidden = teacher_outputs.hidden_states[-1]  # last hidden layer
            teacher_logits = teacher_outputs.logits  # correct logits

        # Student forward pass
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        student_hidden = student_outputs['hidden_states'][-1]
        student_logits = student_outputs['logits']

        # Calculate distillation loss
        loss, hidden_loss_val, logit_loss_val = distillation_loss_fn(
            student_hidden, teacher_hidden,
            student_logits, teacher_logits
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Loss: {loss.item():.4f} | Hidden Loss: {hidden_loss_val:.4f} | Logit Loss: {logit_loss_val:.4f}')
