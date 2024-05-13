## Task-specific fine-tuning script for "LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language"

import os
import sys
import torch
import numpy as np
import datasets
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, LlamaForCausalLM #LLaMAForCausalLM
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from typing import Dict, Union, TypedDict, Optional

MICRO_BATCH_SIZE = 4  
BATCH_SIZE = 128 
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  
LEARNING_RATE = 3e-4 
CUTOFF_LEN = 256 
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
OUTPUT_DIR = "LlamaTurk-7b-t"

config = AutoConfig.from_pretrained("huggyllama/llama-7b")

load_in_4bit=False
load_in_8bit=True

quantization_config: Optional[BitsAndBytesConfig] = BitsAndBytesConfig(
	load_in_4bit=load_in_4bit,
	load_in_8bit=load_in_8bit,
	llm_int8_threshold=6.0,
	llm_int8_has_fp16_weight=False,
	bnb_4bit_compute_dtype=torch.float16,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_type='nf4',
	) if load_in_4bit or load_in_8bit else None

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
	"huggyllama/llama-7b",
	config=config,
	load_in_4bit=load_in_4bit,
	load_in_8bit=load_in_8bit,
	device_map="auto",
	quantization_config=quantization_config,
	torch_dtype=torch.float16,
	)

tokenizer = AutoTokenizer.from_pretrained(
	"huggyllama/llama-7b",
	add_eos_token=True,
	)
	
model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
tokenizer.pad_token_id = 0  

data = datasets.load_dataset("maydogan/TRSAv1", split="train")
data_pos = data.filter(lambda example: example['score'] == "Positive")
data_neg = data.filter(lambda example: example['score'] == "Negative")
data_pos = data_pos.select(range(2500))
data_neg = data_neg.select(range(2500))
data = concatenate_datasets([data_pos, data_neg])

def modify_label(data_point):
	if data_point["score"] == "Positive":
	    data_point["score"] = "olumlu"
	elif data_point["score"] == "Negative":
	    data_point["score"] = "olumsuz"
	return data_point
data = data.map(modify_label)

def generate_prompt(data_point):
	prompt = f"""Aşağıda bir görevi açıklayan talimat bulunmaktadır. Talimatı yeterince sağlayan bir çıktı yaz.

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum: 
{data_point["review"]}

### Çıktı:
{data_point["score"]}"""
	#print(prompt)
	return prompt

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }

data = data.shuffle().map(lambda x: tokenize(generate_prompt(x)))

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False 
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(OUTPUT_DIR)


