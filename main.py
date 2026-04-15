import os
import argparse
import nltk
from dotenv import load_dotenv

# Internal module imports
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

# Import extra credit modules safely
try:
    from src.evaluation.evaluator import Evaluator
except ImportError:
    Evaluator = None

try:
    from src.ui.app import PredictorUI
except ImportError:
    PredictorUI = None

def main():
    # 1. Load configuration
    load_dotenv('config/.env')
    
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    # 2. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Sherlock Holmes N-Gram Predictor")
    parser.add_argument('--step', 
                        choices=['dataprep', 'model', 'inference', 'eval', 'ui', 'all'], 
                        required=True,
                        help="Select the pipeline step to run.")
    args = parser.parse_args()

    # 3. Instantiate objects (No Global Variables)
    normalizer = Normalizer()
    ngram_model = NGramModel(order=int(os.getenv('NGRAM_ORDER', 4)))

    # --- EXECUTION SEQUENCE ---

    # Module 1: Data Preparation
    if args.step in ['dataprep', 'all']:
        print("--- Running Data Preparation ---")
        raw_text = normalizer.load(os.getenv('TRAIN_RAW_DIR'))
        if raw_text:
            stripped_text = normalizer.strip_gutenberg(raw_text)
            sentences = normalizer.sentence_tokenize(stripped_text)
            normalizer.save(sentences, os.getenv('TRAIN_TOKENS'))
            print("Data Prep Complete.")
        else:
            print("Error: No training data found.")

    # Module 2: Model Training
    if args.step in ['model', 'all']:
        print("--- Running Model Training ---")
        token_path = os.getenv('TRAIN_TOKENS')
        if os.path.exists(token_path):
            ngram_model.build_vocab(token_path, int(os.getenv('UNK_THRESHOLD', 3)))
            ngram_model.build_counts_and_probabilities(token_path)
            ngram_model.save_model(os.getenv('MODEL'))
            ngram_model.save_vocab(os.getenv('VOCAB'))
            print("Model Training Complete.")
        else:
            print("Error: Tokens file missing. Run dataprep first.")

   # Module 3: Inference (CLI Loop)
    # MODIFIED: Now runs if step is 'inference' OR 'all'
    if args.step in ['inference', 'all']:
        print("\n--- Entering Interactive Inference Mode ---")
        ngram_model.load(os.getenv('MODEL'), os.getenv('VOCAB'))
        predictor = Predictor(model=ngram_model, normalizer=normalizer)
        
        k = int(os.getenv('TOP_K', 3))
        print(f"Model loaded. Predicting top {k} words.")
        print("Type your phrase (Type 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye.")
                    break
                
                if not user_input:
                    continue

                predictions = predictor.predict_next(user_input, k)
                print(f"Suggestions: {predictions}")
                
            except KeyboardInterrupt:
                break

    # EXTRA CREDIT: Evaluator
    if args.step == 'eval':
        if Evaluator:
            ngram_model.load(os.getenv('MODEL'), os.getenv('VOCAB'))
            evaluator = Evaluator(model=ngram_model, normalizer=normalizer)
            evaluator.run()
        else:
            print("Evaluator module not found.")

    # EXTRA CREDIT: UI
    # Note: 'all' usually doesn't trigger UI because UI requires a different 
    # startup command (streamlit run). We keep this separate.
    if args.step == 'ui':
        if PredictorUI:
            ngram_model.load(os.getenv('MODEL'), os.getenv('VOCAB'))
            predictor = Predictor(model=ngram_model, normalizer=normalizer)
            ui = PredictorUI(predictor=predictor)
            ui.run()
        else:
            print("UI module (Streamlit) not found.")

if __name__ == "__main__":
    main()