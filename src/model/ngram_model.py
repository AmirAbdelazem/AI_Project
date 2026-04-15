import json
import os
from collections import Counter, defaultdict

class NGramModel:
    def __init__(self, order):
        self.order = order
        self.vocab = set()
        self.model = {}

    def build_vocab(self, token_file, threshold):
        counts = Counter()
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f: counts.update(line.split())
        self.vocab = {w for w, c in counts.items() if c >= threshold}
        self.vocab.add("<UNK>")

    def build_counts_and_probabilities(self, token_file):
        raw_counts = {i: defaultdict(Counter) for i in range(1, self.order + 1)}
        total_words = 0
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = [w if w in self.vocab else "<UNK>" for w in line.split()]
                total_words += len(tokens)
                for n in range(1, self.order + 1):
                    for i in range(len(tokens) - n + 1):
                        ngram = tokens[i:i+n]
                        prefix = " ".join(ngram[:-1])
                        raw_counts[n][prefix][ngram[-1]] += 1

        for n in range(1, self.order + 1):
            key = f"{n}gram"
            self.model[key] = {}
            for pref, targets in raw_counts[n].items():
                denom = sum(targets.values()) if n > 1 else total_words
                self.model[key][pref] = {w: c/denom for w, c in targets.items()}

    def lookup(self, context_list):
        for i in range(min(len(context_list), self.order - 1), -1, -1):
            key = f"{i+1}gram"
            pref = " ".join(context_list[len(context_list)-i:]) if i > 0 else ""
            if key in self.model and pref in self.model[key]:
                return self.model[key][pref]
        return {}

    def save_model(self, path):
         os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: json.dump(self.model, f)
    def save_vocab(self, path):
         os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f: json.dump(list(self.vocab), f)
    def load(self, model_path, vocab_path):
        with open(model_path, 'r', encoding='utf-8') as f: self.model = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f: self.vocab = set(json.load(f))