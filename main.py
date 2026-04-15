import os
import argparse
import nltk
from dotenv import load_dotenv

# Absolute imports for clarity
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    load_dotenv('config/.env')
    
    # Required for the normalizer to work
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['dataprep', 'model', 'inference', 'ui', 'all'], required=True)
    args = parser.parse_args()

    # Dependency Injection
    normalizer = Normalizer()
    ngram_model = NGramModel(order=int(os.getenv('NGRAM_ORDER', 4)))

    # --- PIPELINE ---
    if args.step in ['dataprep', 'all']:
        raw = normalizer.load(os.getenv('TRAIN_RAW_DIR'))
        clean = normalizer.strip_gutenberg(raw)
        sentences = normalizer.sentence_tokenize(clean)
        normalizer.save(sentences, os.getenv('TRAIN_TOKENS'))
        print("Done: Data Prepared.")

    if args.step in ['model', 'all']:
        token_path = os.getenv('TRAIN_TOKENS')
        ngram_model.build_vocab(token_path, int(os.getenv('UNK_THRESHOLD', 3)))
        ngram_model.build_counts_and_probabilities(token_path)
        ngram_model.save_model(os.getenv('MODEL'))
        ngram_model.save_vocab(os.getenv('VOCAB'))
        print("Done: Model Trained.")

    if args.step == 'inference':
        ngram_model.load(os.getenv('MODEL'), os.getenv('VOCAB'))
        predictor = Predictor(model=ngram_model, normalizer=normalizer)
        while True:
            text = input("Query > ").strip()
            if text.lower() == 'exit': break
            print(predictor.predict_next(text, 3))

    if args.step == 'ui':
        # 1. Prepare data for the UI
        ngram_model.load(os.getenv('MODEL'), os.getenv('VOCAB'))
        predictor = Predictor(model=ngram_model, normalizer=normalizer)
        
        # 2. Launch UI
        from src.ui.app import PredictorUI
        ui = PredictorUI(predictor)
        ui.run()

if __name__ == "__main__":
    main()