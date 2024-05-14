# LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language

[Paper](https://arxiv.org/abs/2405.07745)

[Published Models (to run PEFT models, see inference code below)](https://huggingface.co/metunlp)

[Instruction set used in instruction fine-tuning](data/llamaturk_instruction_set.json)

[Instruction set used in instruction fine-tuning (at HuggingFace)](https://huggingface.co/datasets/metunlp/LlamaTurk-Instruction-Set)

[Dataset used in tasks-specific fine-tuning](https://huggingface.co/datasets/maydogan/TRSAv1)

[Evaluation benchmark dataset: Belebele](https://huggingface.co/datasets/facebook/belebele/viewer/default/tur_Latn)

[Evaluation benchmark dataset: XCOPA](https://huggingface.co/datasets/xcopa/viewer/tr)

[Source code for instruction fine-tuning](src/finetune_instruction.py)

[Source code for task-specific fine-tuning](src/finetune_task.py)

[Source code for continual training](src/continual_train.py)

[Source code for vocabulary extension](src/vocabulary_extension.py)

[Source code for instrinsic evaluation (perplexity calculation)](src/perplexity.py)

[Source code for extrinsic evaluation (task inference)](src/inference_task.py)

[Source code for getting model checkpoints for an adapted model](src/merge_base_and_finetuned_models.py)

Citation:
```
@misc{toraman2024llamaturk,
      title={LlamaTurk: Adapting Open-Source Generative Large Language Models for Low-Resource Language}, 
      author={Cagri Toraman},
      year={2024},
      eprint={2405.07745},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
