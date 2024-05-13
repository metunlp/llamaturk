## Getting final model checkpoints by merging base model and fine-tuned LoRA model for "LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language"

import torch
import fire
from peft import PeftModel
from transformers import LlamaForCausalLM

base_model="huggyllama/llama-7b"
lora_model="LlamaTurk-7b-i"
save_path="LlamaTurk-7b-i-checkpoints"

model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
)

model = PeftModel.from_pretrained(
        model,
        lora_model,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(save_path)
