from collections import Counter
from dataclasses import dataclass

from src.cleaning_preprocessing import LanguageData


@dataclass
class LanguageFeatures:
    char_ngrams: Counter[str]
    word_ngrams: Counter[str]
    avg_word_len: float
    avg_sent_len: float


class LanguageFeatureExtractor:
    """
    Extracts features from a LanguageData object:
    - Character n-grams (with start/end padding)
    - Word n-grams
    """

    def __init__(self, lang_data: LanguageData):
        self.lang_data = lang_data
        self.char_ngrams: Counter | None = None
        self.word_ngrams: Counter | None = None

    # -----------------------------
    # MAIN METHODS
    # -----------------------------

    def char_ngram(self, n: int = 3, start: str = "_", end: str = "_") -> Counter:
        """Compute character n-grams frequency"""
        ngram_counts = Counter()
        for sentence in self.lang_data.sentences:
            padded = f"{start}{sentence}{end}"
            if len(padded) >= n:
                ngram_counts.update(
                    [padded[i : i + n] for i in range(len(padded) - n + 1)]
                )
        self.char_ngrams = ngram_counts
        return ngram_counts

    def word_ngram(self, n: int = 1) -> Counter[str]:
        """Compute word n-grams frequency"""
        ngram_counts = Counter()
        for sentence in self.lang_data.sentences:
            words = sentence.split()
            if len(words) >= n:
                ngram_counts.update(
                    [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
                )
        self.word_ngrams = ngram_counts
        return ngram_counts

    def get_features(self) -> LanguageFeatures:
        """
        Returns all extracted language features as a LanguageFeatures dataclass.
        """
        if not isinstance(self.char_ngrams, Counter):
            raise TypeError(
                f"char_ngrams must be a Counter. Current type: {type(self.char_ngrams)}"
            )

        if not isinstance(self.word_ngrams, Counter):
            raise TypeError(
                f"word_ngrams must be a Counter. Current type: {type(self.word_ngrams)}"
            )

        if not isinstance(self.lang_data.avg_word_len, float):
            raise TypeError(
                f"avg_word_len must be a float. "
                f"Current type: {type(self.lang_data.avg_word_len)}"
            )

        if not isinstance(self.lang_data.avg_sent_len, float):
            raise TypeError(
                f"avg_sent_len must be a float. "
                f"Current type: {type(self.lang_data.avg_sent_len)}"
            )

        return LanguageFeatures(
            char_ngrams=self.char_ngrams,
            word_ngrams=self.word_ngrams,
            avg_word_len=self.lang_data.avg_word_len,
            avg_sent_len=self.lang_data.avg_sent_len,
        )

    # -----------------------------
    # EXTERNAL HELPER METHODS
    # -----------------------------

    def top_ngrams(self, n: int = 20, kind: str = "char") -> list[tuple[str, int]]:
        """Return the top `n` most frequent n-grams."""
        if kind == "char":
            if self.char_ngrams is None:
                raise ValueError(
                    "Character n-grams not computed yet. Call char_ngram() first."
                )
            return self.char_ngrams.most_common(n)
        elif kind == "word":
            if self.word_ngrams is None:
                raise ValueError(
                    "Word n-grams not computed yet. Call word_ngram() first."
                )
            return self.word_ngrams.most_common(n)
        else:
            raise ValueError("Invalid kind. Use 'char' or 'word'.")

    def summary(self, n: int = 20) -> None:
        """Print a summary of the language features with top-n n-grams."""
        print(f"Language: {self.lang_data.name}")
        print(f"Avg. word length: {self.lang_data.avg_word_len:.2f}")
        print(f"Avg. sentence length: {self.lang_data.avg_sent_len:.2f}\n")

        print(f"Top {n} character n-grams:")
        for gram, count in self.top_ngrams(n=n, kind="char"):
            print(f"{gram}: {count}")

        print(f"\nTop {n} word n-grams:")
        for gram, count in self.top_ngrams(n=n, kind="word"):
            print(f"{gram}: {count}")
