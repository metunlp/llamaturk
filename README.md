<p align="center"><img src="llamaturk.png" width="300"></p>
<p align="center"><img src="odtu_metu.png" width="600"></p>

# LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language

[Paper](https://arxiv.org/abs/2405.07745)

[Published Models (to run PEFT models, see inference code below)](https://huggingface.co/metunlp)

[Instruction set used in instruction fine-tuning](data/llamaturk_instruction_set.json)

[Instruction set used in instruction fine-tuning (at HuggingFace)](https://huggingface.co/datasets/metunlp/LlamaTurk-Instruction-Set)

[Dataset used in tasks-specific fine-tuning](https://huggingface.co/datasets/maydogan/TRSAv1)

[Evaluation benchmark dataset: Belebele](https://huggingface.co/datasets/facebook/belebele/viewer/default/tur_Latn)

[Evaluation benchmark dataset: XCOPA](https://huggingface.co/datasets/xcopa/viewer/tr)

[Tokenizer used in vocabulary extension](bpe_28k_oscar_tr_tokenizer.json)

[Source code for instruction fine-tuning](src/finetune_instruction.py)

[Source code for task-specific fine-tuning](src/finetune_task.py)

[Source code for continual training](src/continual_train.py)

[Source code for vocabulary extension](src/vocabulary_extension.py)

[Source code for instrinsic evaluation (perplexity calculation)](src/perplexity.py)

[Source code for extrinsic evaluation (task inference)](src/inference_task.py)

[Source code for getting model checkpoints for an adapted model](src/merge_base_and_finetuned_models.py)

Citation:
```
@inproceedings{toraman-2024-adapting,
    title = "Adapting Open-Source Generative Large Language Models for Low-Resource Languages: A Case Study for {T}urkish",
    author = "Toraman, Cagri",
    booktitle = "Proceedings of the Fourth Workshop on Multilingual Representation Learning (MRL 2024)",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.mrl-1.3",
    doi = "10.18653/v1/2024.mrl-1.3",
    pages = "30--44",
}
```
