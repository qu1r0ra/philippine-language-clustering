import logging
import os

import regex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageData:
    """
    Represents a single language corpus with preprocessed text
    and basic linguistic metrics.
    """

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.sentences: list[str] = []
        self.avg_word_len: float | None = None
        self.avg_sent_len: float | None = None
        self._loaded = False

    # -----------------------------
    # MAIN METHODS
    # -----------------------------

    def load(self):
        """
        Loads and normalizes all .txt files for this language corpus.
        Automatically computes for basic statistics.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Directory not found: {self.path}")

        self.sentences = self._load_corpus()
        self.avg_word_len = self._compute_avg_word_length()
        self.avg_sent_len = self._compute_avg_sentence_length()

        return self

    # -----------------------------
    # INTERNAL HELPER METHODS
    # -----------------------------

    def _load_corpus(self):
        sentences = []
        for file_name in os.listdir(self.path):
            if file_name.endswith(".txt"):
                with open(os.path.join(self.path, file_name)) as file:
                    lines = [line.strip() for line in file if line.strip()]
                    sentences.extend([self._normalize(line) for line in lines])
        return sentences

    def _normalize(self, text: str) -> str:
        """
        Normalizes a string by lowercasing it and removing punctuation, digits,
        and extra spaces.
        """
        text = text.lower()
        text = regex.sub(r"[^\p{L}\s]", "", text)  # remove non-Unicode characters
        text = regex.sub(r"\s+", " ", text).strip()  # remove extra spaces
        return text

    def _compute_avg_word_length(self) -> float:
        """Computes average number of characters per word."""
        words = [w for s in self.sentences for w in s.split()]
        return sum(len(w) for w in words) / len(words) if words else 0

    def _compute_avg_sentence_length(self) -> float:
        """Computes average number of words per sentence."""
        word_counts = [len(s.split()) for s in self.sentences]
        return sum(word_counts) / len(word_counts) if word_counts else 0

    # -----------------------------
    # EXTERNAL HELPER METHODS
    # -----------------------------

    def summary(self) -> None:
        """Print a summary of the language corpus."""
        if not self.sentences:
            logging.info(f"LanguageData({self.name}): not loaded. Call .load() first.")

        logging.info(f"Language: {self.name}")
        logging.info(f"No. of sentences: {len(self.sentences)}")
        logging.info(f"Avg. word length: {self.avg_word_len:.2f}")
        logging.info(f"Avg. sentence length: {self.avg_sent_len:.2f}")
