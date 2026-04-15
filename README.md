# N-Gram Predictor

A Sherlock Holmes-themed next-word predictor built using N-gram Maximum Likelihood Estimation (MLE) and Stupid Backoff.

## Requirements
- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

## Setup
1. Clone the repository.
2. Create `config/.env` using the provided template.
3. Place training books (.txt) in `data/raw/train/`.

## Usage
Run the full pipeline:
```bash
python main.py --step all