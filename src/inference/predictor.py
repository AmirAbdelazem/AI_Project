class Predictor:
    def __init__(self, model, normalizer):
        self.model = model
        self.normalizer = normalizer

    def predict_next(self, text, k):
        # 1. Clean the input just like the training data
        clean_text = self.normalizer.normalize(text)
        tokens = self.normalizer.word_tokenize(clean_text)
        
        # 2. Get the context (last N-1 words)
        context = tokens[-(self.model.order - 1):] if len(tokens) >= self.model.order else tokens
        
        # 3. Get probabilities from model (handles backoff internally)
        probs = self.model.lookup(context)
        
        # 4. Sort and return top K
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_probs[:k]]