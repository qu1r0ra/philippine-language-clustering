from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.cleaning_preprocessing import LanguageData

NgramKind = Literal["char", "word"]


@dataclass
class LanguageFeatures:
    lang_data: LanguageData
    char_ngrams: Counter[str]
    word_ngrams: Counter[str]


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

        return LanguageFeatures(
            lang_data=self.lang_data,
            char_ngrams=self.char_ngrams,
            word_ngrams=self.word_ngrams,
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


class FeatureSpaceBuilder:
    """
    Builds global feature spaces (character and word n-gram vocabularies)
    from a collection of LanguageFeatureExtractor objects.

    Responsibilities:
    - Construct global vocabularies for char and word n-grams.
    - Convert per-language Counters into aligned, normalized vectors.
    - Provide matrix outputs for later similarity or clustering steps.
    """

    def __init__(self, feature_extractors: dict[str, LanguageFeatureExtractor]):
        self.feature_extractors = feature_extractors

        # global vocabularies
        self.char_vocab: list[str] = []
        self.word_vocab: list[str] = []

        # per-language feature vectors
        self.char_vectors: dict[str, np.ndarray] = {}
        self.word_vectors: dict[str, np.ndarray] = {}

    # -----------------------------
    # MAIN METHODS
    # -----------------------------

    def build_feature_space(self) -> None:
        """Wrapper method that calls builder methods to build the feature space."""
        self.build_global_vocab()
        self.build_vectors()

    def build_global_vocab(self) -> None:
        """Collect all unique n-grams across languages into global vocabularies."""
        char_ngrams_set, word_ngrams_set = set(), set()

        for extractor in self.feature_extractors.values():
            if extractor.char_ngrams:
                char_ngrams_set.update(extractor.char_ngrams.keys())
            if extractor.word_ngrams:
                word_ngrams_set.update(extractor.word_ngrams.keys())

        self.char_vocab = sorted(char_ngrams_set)
        self.word_vocab = sorted(word_ngrams_set)

        print(
            f"Built global vocabularies: "
            f"{len(self.char_vocab)} char n-grams, {len(self.word_vocab)} word n-grams."
        )

    def build_vectors(self) -> None:
        """Build normalized frequency vectors for all languages."""
        if not self.char_vocab or not self.word_vocab:
            raise RuntimeError("Call build_global_vocab() before build_vectors().")

        # Construct char and word vectors
        for lang, extractor in self.feature_extractors.items():
            if extractor.char_ngrams and extractor.word_ngrams:
                total_char_ngrams = sum(extractor.char_ngrams.values())
                char_vec = np.array(
                    [
                        extractor.char_ngrams.get(ngram, 0) / total_char_ngrams
                        for ngram in self.char_vocab
                    ],
                    dtype=float,
                )
                self.char_vectors[lang] = char_vec

                total_word_ngrams = sum(extractor.word_ngrams.values())
                word_vec = np.array(
                    [
                        extractor.word_ngrams.get(ngram, 0) / total_word_ngrams
                        for ngram in self.word_vocab
                    ],
                    dtype=float,
                )
                self.word_vectors[lang] = word_vec

        print(
            f"Constructed normalized vectors for {len(self.feature_extractors)} "
            f"languages."
        )

    # -----------------------------
    # EXTERNAL HELPER METHODS
    # -----------------------------

    def get_vector(self, lang: str, kind: NgramKind = "char") -> pd.Series:
        """
        Retrieve the normalized vector for a given language as a pandas Series.

        Returns:
            pd.Series: Indexed by n-gram (global vocabulary), containing normalized
            frequencies.
        """
        vec_dict = self.char_vectors if kind == "char" else self.word_vectors

        if lang not in vec_dict:
            raise KeyError(
                f"Language '{lang}' not found or {kind}-vector not computed."
            )

        lang_vec = vec_dict[lang]

        vocab = self.char_vocab if kind == "char" else self.word_vocab

        return pd.Series(lang_vec, index=vocab, name=lang)

    def as_matrix(self, kind: NgramKind = "char") -> pd.DataFrame:
        """
        Return the feature matrix for all languages as a pandas DataFrame.

        Indexes = language names
        Columns = global vocabulary (n-grams)
        Values = normalized frequencies
        """
        vec_dict = self.char_vectors if kind == "char" else self.word_vectors

        if not vec_dict:
            raise ValueError(
                f"No {kind}-vectors have been computed yet. "
                f"Call build_feature_space(kind='{kind}') first."
            )

        vocab = self.char_vocab if kind == "char" else self.word_vocab

        df = pd.DataFrame.from_dict(
            vec_dict, orient="index", columns=vocab
        ).sort_index()

        return df
