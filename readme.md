# Facet-Generation

This is the source code for paper [A Comparative Study of Training Objectives for Clarification Facet Generation(SIGIR-AP 2023)](https://arxiv.org/pdf/2310.00703v1.pdf)

# Requirements and Installation

- Python version >= 3.8
- PyTorch version >= 1.11.0
- Transformers version >= 4.27.2

# Corpus

- MIMICS: [microsoft/MIMICS: MIMICS: A Large-Scale Data Collection for Search Clarification (github.com)](https://github.com/microsoft/MIMICS)
  - You should download `The Bing API's Search Results for MIMICS Queries`

# Models

We just provide model weights in the following websites(`Seq-default` contains both model weights and the tokenizer). The tokenizer for each model is the same with the tokenizer used in `Bart-base`(https://huggingface.co/facebook/bart-base)  

- `seq-default`: https://huggingface.co/algoprog/mimics-bart-base
- `seq-min-perm`: https://huggingface.co/Shiyunee/seq-min-perm
- `seq-avg-perm`: https://huggingface.co/Shiyunee/seq-avg-perm
- `set-pred`: https://huggingface.co/Shiyunee/set-pred
- `seq-set-pred`: https://huggingface.co/Shiyunee/seq-set-pred

# Usage

- `data_process.py`: prepare data for inference
- `inference.py`: generate facets for the given data
- `evaluation.py`: evaluate the results
- `score.py`: replace `score.py` in `bert_score`(package) with this file so we can load the model and tokenizer only once