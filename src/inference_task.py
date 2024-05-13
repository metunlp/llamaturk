## Task-specific inference script for "LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language"

import os
import re
import torch
import transformers
import datasets
import pandas as pd
import random
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, PreTrainedModel, LlamaForCausalLM
from peft import PeftModel, PeftConfig
from typing import Dict, Union, TypedDict, Optional

BASE_MODEL = "huggyllama/llama-7b"
FINETUNED = True #make it True to use fine-tuned LlamaTurk models.
FINETUNED_MODEL = "LlamaTurk-7b-i"
SHOT = "zero-shot"

if FINETUNED:	
	config = PeftConfig.from_pretrained(FINETUNED_MODEL) 
else:
	config = AutoConfig.from_pretrained(BASE_MODEL)

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
	BASE_MODEL,
	config=config,
	load_in_4bit=load_in_4bit,
	load_in_8bit=load_in_8bit,
	device_map="auto",
	quantization_config=quantization_config,
	torch_dtype=torch.float16,
	).eval()

tokenizer = AutoTokenizer.from_pretrained(
	BASE_MODEL,	
	)

if FINETUNED:		
	model = PeftModel.from_pretrained(model, FINETUNED_MODEL).eval() 

def generate_prompt(instruction):
    if SHOT == "zero-shot":
        prompt = f"""[INST]Aşağıda bir görevi açıklayan talimat bulunmaktadır. Talimatı yeterince sağlayan bir çıktı yaz.

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum: 
{instruction}[/INST]

### Çıktı:
"""
    elif SHOT == "1-shot":
        prompt = f"""[INST]Aşağıda bir görevi açıklayan talimat bulunmaktadır. Talimatı yeterince sağlayan bir çıktı yaz.

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:  
çok güzel, sağlıklı, temiz, ferah

### Çıktı:
olumlu

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum: 
{instruction}[/INST]

### Çıktı:
"""
    elif SHOT == "2-shot":
        prompt = f"""[INST]Aşağıda bir görevi açıklayan talimat bulunmaktadır. Talimatı yeterince sağlayan bir çıktı yaz.

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:  
çok güzel, sağlıklı, temiz, ferah

### Çıktı:
olumlu

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:  
hiç bir işe yaramıyor

### Çıktı:
olumsuz

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:  
{instruction}[/INST]

### Çıktı:
"""
    elif SHOT == "3-shot":
        prompt = f"""[INST]Aşağıda bir görevi açıklayan talimat bulunmaktadır. Talimatı yeterince sağlayan bir çıktı yaz.

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:  
çok güzel, sağlıklı, temiz, ferah

### Çıktı:
olumlu

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:  
hiç bir işe yaramıyor

### Çıktı:
olumsuz

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:
Çok güzel fiyatı da gayet uygun

### Çıktı:
olumlu

### Talimat:
Lütfen verilen yorumun olumlu ya da olumsuz olduğunu çıktı olarak belirtin.

### Yorum:
{instruction}[/INST]

### Çıktı:
"""
    return prompt

generation_config = GenerationConfig(
	temperature=0.2,
	top_p=0.75,
	num_beams=4,
	return_dict_in_generate=True,
	output_scores=True,
	max_new_tokens=256,
	#no_repeat_ngram_size=4,
	repetition_penalty=1.8,
	
)

def evaluate(instruction):
    prompt = generate_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    generation_output = model.generate(
		input_ids=input_ids,
		generation_config=generation_config,
	)
    result=""
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        result = output.split("### Çıktı:")[-1].strip()	
    return prompt, result

hit_pos = 0
hit_neg = 0
total = 0
pos = 0
neg = 0
dataset = datasets.load_dataset("maydogan/TRSAv1")["train"]
for i, row in reversed(list(enumerate(dataset))):
        if i > 149980:
            continue
        if (pos == 50 and row["score"] == "Positive") or (neg == 50 and row["score"] == "Negative") or (row["score"] == "Neutral"): 
            continue
        if row["score"] == "Positive":
            pos = pos + 1
            label_str = "olumlu"
        elif row["score"] == "Negative":
            neg = neg + 1
            label_str = "olumsuz"
        prompt, response = evaluate(row["review"])
        if len(response) > 5:
            if response.strip()[:6].lower() == "olumlu":
                print("found olumlu")
                response = "olumlu" 		
            elif response.strip()[:7].lower() == "olumsuz":
                print("found olumsuz")
                response = "olumsuz"
        if label_str == response and label_str == "olumlu":
            hit_pos = hit_pos + 1            
        elif label_str == response and label_str == "olumsuz":
            hit_neg = hit_neg + 1            
        total = total + 
        if (pos + neg) == 100:
            break

print("overall accuracy: ", (hit_pos+hit_neg)/total)