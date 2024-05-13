## Perplexity calculation script for "LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language"

import datasets
import os
import torch
import random
import pandas as pd
from evaluate import load
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, PreTrainedModel, LlamaForCausalLM
from typing import Dict, Union, TypedDict, Optional
from peft import PeftModel, PeftConfig
from tokenizers import Tokenizer

BASE_MODEL = "huggyllama/llama-7b"
FINETUNED = True #make it True to use fine-tuned LlamaTurk models.
FINETUNED_MODEL = "LlamaTurk-7b-i"
VOCAB_EXTENSION = True
DATA = "xquad-question"

if FINETUNED = True:	
	config = PeftConfig.from_pretrained(FINETUNED_MODEL) 
else:
	config = AutoConfig.from_pretrained(BASE_MODEL)

load_in_4bit=False
load_in_8bit=True

quantization_config: Optional[BitsAndBytesConfig] = BitsAndBytesConfig(load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4') if load_in_4bit or load_in_8bit else None

model: PreTrainedModel = AutoModelForCausalLM.from_pretrained( BASE_MODEL, config=config, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16, trust_remote_code=True, use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, batched=True)

if VOCAB_EXTENSION == True:
	tokenizer2 = Tokenizer.from_file("bpe_28k_oscar_tr_tokenizer.json")
	tokens_differing = set(tokenizer2.get_vocab()).difference(tokenizer.get_vocab())
	tokenizer.add_tokens(list(tokens_differing))
	model.resize_token_embeddings(len(tokenizer))

if FINETUNED == True:		
	model = PeftModel.from_pretrained(model, FINETUNED_MODEL) 

maxcharlength = 5000000
if DATA == "xquad-question":
	input_texts = datasets.load_dataset("xquad", "xquad.tr", split="validation")
	column_name = "question"
elif DATA == "xquad-context":
	input_texts = datasets.load_dataset("xquad", "xquad.tr", split="validation")
	column_name = "context"
elif DATA == "databricks-instruction":
	input_texts = datasets.load_dataset("atasoglu/databricks-dolly-15k-tr", split="train")
	column_name = "instruction"
elif DATA == "databricks-response":
	input_texts = datasets.load_dataset("atasoglu/databricks-dolly-15k-tr", split="train")
	column_name = "response"

full_text = "\n\n".join(input_texts[column_name])
if len(full_text) > maxcharlength:
	full_text = full_text[0:maxcharlength]
encodings = tokenizer(full_text, return_tensors="pt")

device = "cuda"
import torch
from tqdm import tqdm

max_length = model.config.max_position_embeddings
stride = 512
seq_len = encodings.input_ids.size(1)
nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("ppl score: ",ppl)