# src/models.py
import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig

class StudentModel(nn.Module):
    def __init__(self, student_config_path=None):
        super(StudentModel, self).__init__()
        if student_config_path:
            config = RobertaConfig.from_json_file(student_config_path)
        else:
            config = RobertaConfig(
                hidden_size=384,
                num_attention_heads=6,
                num_hidden_layers=6,
                intermediate_size=1536,
                vocab_size=50265,  # same as CodeBERT
                max_position_embeddings=514,
            )
        self.config = config

        # Encoder
        self.student_encoder = RobertaModel(config)

        # Simple MLM head
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        encoder_outputs = self.student_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = encoder_outputs.hidden_states[-1]
        logits = self.mlm_head(last_hidden_state)  # predict vocab tokens

        if return_dict:
            return {
                "logits": logits,
                "hidden_states": encoder_outputs.hidden_states
            }
        else:
            return logits, encoder_outputs.hidden_states
