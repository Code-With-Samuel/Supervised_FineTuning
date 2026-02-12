# Supervised Fine-Tuning (SFT)

This repository contains a minimal example for fine-tuning `bert-base-uncased` on the IMDB sentiment dataset using Hugging Face Transformers and `datasets`.

## Contents
- `bert-base-uncased_finetuning.ipynb` — Jupyter notebook that prepares the dataset, fine-tunes the model, and evaluates it.
- `results/` — Checkpoints and training outputs (many checkpoints saved during training).

## Requirements
- Python 3.8+
- torch
- transformers
- datasets

Install dependencies (recommended inside a virtual environment):

```bash
pip install torch transformers datasets
```

## Quick usage

1. Open `bert-base-uncased_finetuning.ipynb` in Jupyter/VS Code.
2. If you already trained the model and have checkpoints in `./results`, you can evaluate without re-training. See the Evaluation cell for how to load a checkpoint (for example `./results/checkpoint-4689`).

Example evaluation snippet (already present in the notebook):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader

# reload tokenizer and dataset (not strictly necessary if already in memory)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# load tokenized dataset or recreate from the IMDB dataset
# tokenized_dataset = ...

# load checkpoint
model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-4689")
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# evaluate using DataLoader over tokenized_dataset['test']
```

## Notes
- The `results/` folder may contain large model files. Use `.gitignore` and Git LFS for `*.safetensors` or other large artifacts.
- If you plan to push model files to GitHub, enable Git LFS and track the large file types before committing:

```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
```

## Pushing to GitHub
If the repository is not yet a git repo, run:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Code-With-Samuel/Supervised_FineTuning.git
git push -u origin main
```

If remote already exists and you want to set the URL:

```bash
git remote set-url origin https://github.com/Code-With-Samuel/Supervised_FineTuning.git
git push -u origin main
```

## License

MIT
