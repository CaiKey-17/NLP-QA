import torch.nn as nn
import torch
from transformers import AutoModel
# ============================
# 📌 MODEL PHOBERT EXTRACTIVE
# ============================
class PhoBERTForQA(nn.Module):
    def __init__(self, name="vinai/phobert-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        H = self.encoder.config.hidden_size
        self.qa_head = nn.Linear(H, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        logits = self.qa_head(last_hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        return {
            "start_logits": start_logits.squeeze(-1),
            "end_logits": end_logits.squeeze(-1),
        }

class ExtractiveQAModel:
    def __init__(self, model, tokenizer, max_length=256):
        self.model = model
        self.tok = tokenizer
        self.max_length = max_length

    def predict_span(self, ctx, ques):
        enc = self.tok(
            ques,
            ctx,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc.pop("token_type_ids", None)
        # enc = {k: v.to(device) for k,v in enc.items()}

        out = self.model(enc["input_ids"], enc["attention_mask"])
        start_logits = out["start_logits"][0]
        end_logits   = out["end_logits"][0]

        # use argmax instead of top-k
        s = torch.argmax(start_logits).item()
        e = torch.argmax(end_logits).item()

        # Ensure span is valid
        if e < s:
            e = s

        # find context start
        sep_idx = (enc["input_ids"][0] == self.tok.sep_token_id).nonzero()[0].item()
        ctx_start = sep_idx + 1

        # Span must be inside context region
        if s < ctx_start or e < ctx_start:
            return ""

        ids = enc["input_ids"][0][s:e+1]
        return self.tok.decode(ids, skip_special_tokens=True)

    def predict(self, ctxs, ques):
        return [self.predict_span(c, q) for c, q in zip(ctxs, ques)]
