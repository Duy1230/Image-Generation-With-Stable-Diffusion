import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        clip_id = "openai/clip-vit-base-patch32"
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_id)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_encoder.eval()
        self.text_encoder.to(self.device)

    def forward(self, prompts):
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.text_encoder.config.max_position_embeddings,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            text_encoder_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        last_hidden_states = text_encoder_output.last_hidden_state

        return last_hidden_states
