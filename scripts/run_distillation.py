# scripts/run_distillation.py
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM
from src.models import StudentModel
from src.train import train

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }

    def __len__(self):
        return len(self.encodings['input_ids'])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load teacher model and tokenizer from local fine-tuned checkpoint
    teacher_model_path = './models/teacher_finetuned/'  # <--- point this to your local path
    tokenizer = RobertaTokenizer.from_pretrained(teacher_model_path)
    teacher_model = RobertaForMaskedLM.from_pretrained(teacher_model_path).to(device)

    student_model = StudentModel().to(device)

    toy_data = [
        "assertEquals(5, add(2, 3));",
        "assertTrue(list.contains(item));",
        "assertNotNull(object);",
    ]
    dataset = ToyDataset(toy_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)

    train(student_model, teacher_model, dataloader, optimizer, device)

if __name__ == "__main__":
    main()
