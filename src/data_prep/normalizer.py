import os
import re
import string
import nltk

class Normalizer:
    """Handles loading, cleaning, tokenizing, and saving the corpus."""

    def load(self, folder_path):
        """Loads all .txt files from the training folder."""
        combined_text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    combined_text += f.read() + "\n"
        return combined_text

    def strip_gutenberg(self, text):
        """Removes Gutenberg headers and footers using specific markers."""
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = text.find(start_marker)
        if start_idx != -1:
            line_end = text.find("***", start_idx + len(start_marker))
            text = text[line_end + 3:]
            
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]
        return text.strip()

    def normalize(self, text):
        """Applies lowercase, strips punctuation/numbers, and fixes whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def sentence_tokenize(self, text):
        """Splits text into sentences."""
        return nltk.sent_tokenize(text)

    def word_tokenize(self, sentence):
        """Splits a sentence into tokens."""
        return sentence.split()

    def save(self, sentences, filepath):
        """Writes normalized sentences to train_tokens.txt."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for sent in sentences:
                clean_sent = self.normalize(sent)
                if clean_sent:
                    f.write(clean_sent + "\n")