import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from safetensors.torch import load_file
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# PHOBERT EXTRACTIVE QA
# ============================
import os

from phobert_helpers import PhoBERTForQA, ExtractiveQAModel


def load_phobert_model(path):
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = PhoBERTForQA()
    state = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return ExtractiveQAModel(model, tok)


# ============================
# ViT5 GENERATIVE QA
# ============================
GEN_KWARGS = {
    "max_new_tokens": 64,
    "do_sample": False,
}

def load_vit5_model(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()

    def run_batch(ctxs, ques):
        outs = []
        for c, q in zip(ctxs, ques):
            inp = f"Ngữ cảnh: {c}\nCâu hỏi: {q}\nTrả lời:"
            enc = tok([inp], return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(**enc, **GEN_KWARGS)
            outs.append(tok.decode(gen[0], skip_special_tokens=True))
        return outs

    return run_batch


# ============================
# Qwen GENERATIVE QA
# ============================
def load_qwen_model(path):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    state = load_file(os.path.join(path, "model.safetensors"))
    model.load_state_dict(state, strict=False)

    model.to(device).eval()

    def run_batch(ctxs, ques):
        outs = []
        for c, q in zip(ctxs, ques):

            prompt = f"Ngữ cảnh: {c}\nCâu hỏi: {q}\nTrả lời:"

            enc = tok([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(**enc, max_new_tokens=64)

            full_output = tok.decode(gen[0], skip_special_tokens=True)

            # ============================
            # CHỈ LẤY PHẦN SAU 'Trả lời:'
            # ============================
            if "Trả lời:" in full_output:
                answer = full_output.split("Trả lời:", 1)[-1].strip()
            else:
                answer = full_output.strip()

            # XÓA NGỮ CẢNH / CÂU HỎI NẾU BỊ LẶP LẠI  
            answer = answer.replace(c, "").strip()
            answer = answer.replace(q, "").strip()

            # Xóa mọi phần vô tình lặp lại
            for key in ["Ngữ cảnh:", "Câu hỏi:", "Trả lời:"]:
                answer = answer.split(key)[0].strip()

            outs.append(answer)

        return outs

    return run_batch

