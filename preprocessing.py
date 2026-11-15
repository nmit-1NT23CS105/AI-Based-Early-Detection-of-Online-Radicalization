from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from nltk.stem import PorterStemmer, WordNetLemmatizer


ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}

HINDI_STOPWORDS = {
    "aur",
    "ek",
    "hai",
    "ham",
    "hum",
    "ka",
    "ki",
    "ke",
    "ko",
    "mein",
    "par",
    "se",
    "tha",
    "thi",
    "the",
    "unka",
    "unke",
    "unkae",
    "yeh",
    "woh",
    "और",
    "एक",
    "का",
    "की",
    "के",
    "को",
    "में",
    "पर",
    "से",
    "यह",
    "वह",
    "है",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:(?:\+?\d{1,3})?[-.\s]?)?(?:\d{10,12})")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s\u0900-\u097F']")
MULTISPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[\u0900-\u097F]+")


@dataclass
class PreprocessedText:
    original_text: str
    cleaned_text: str
    normalized_text: str
    tokens: List[str]
    anonymized_text: str
    language: str
    input_hash: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


class NLPPreprocessor:
    def __init__(self, use_lemmatization: bool = True) -> None:
        self.use_lemmatization = use_lemmatization
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None

    def detect_language(self, text: str) -> str:
        devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        if devanagari_count and latin_count:
            return "mixed"
        if devanagari_count:
            return "hindi"
        return "english"

    def anonymize(self, text: str) -> str:
        masked = URL_PATTERN.sub("[LINK]", text)
        masked = EMAIL_PATTERN.sub("[EMAIL]", masked)
        masked = PHONE_PATTERN.sub("[PHONE]", masked)
        masked = NUMBER_PATTERN.sub("[NUM]", masked)
        masked = re.sub(r"@\w+", "@user", masked)
        return MULTISPACE_PATTERN.sub(" ", masked).strip()

    def clean_text(self, text: str) -> str:
        text = text or ""
        text = html.unescape(text)
        text = text.replace("\n", " ").replace("\r", " ")
        text = URL_PATTERN.sub(" ", text)
        text = EMAIL_PATTERN.sub(" ", text)
        text = PHONE_PATTERN.sub(" ", text)
        text = PUNCTUATION_PATTERN.sub(" ", text)
        text = MULTISPACE_PATTERN.sub(" ", text).strip().lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def normalize_token(self, token: str) -> str:
        if not token:
            return token
        normalized = token.lower()
        if normalized in ENGLISH_STOPWORDS or normalized in HINDI_STOPWORDS:
            return ""
        if self.lemmatizer is not None and normalized.isascii():
            try:
                normalized = self.lemmatizer.lemmatize(normalized)
            except LookupError:
                pass
        if normalized.isascii():
            normalized = self.stemmer.stem(normalized)
        return normalized

    def preprocess(self, text: str) -> PreprocessedText:
        original = (text or "").strip()
        cleaned = self.clean_text(original)
        raw_tokens = self.tokenize(cleaned)
        tokens = [token for token in (self.normalize_token(token) for token in raw_tokens) if token]
        normalized_text = " ".join(tokens)
        anonymized_text = self.anonymize(original)
        language = self.detect_language(original)
        input_hash = hashlib.sha256(anonymized_text.encode("utf-8")).hexdigest()
        return PreprocessedText(
            original_text=original,
            cleaned_text=cleaned,
            normalized_text=normalized_text,
            tokens=tokens,
            anonymized_text=anonymized_text,
            language=language,
            input_hash=input_hash,
        )


class EmotionAnalyzer:
    anger_terms = {
        "anger",
        "angry",
        "attack",
        "betray",
        "destroy",
        "furious",
        "hate",
        "hostile",
        "rage",
        "retaliate",
        "revenge",
        "traitor",
        "उग्र",
        "गुस्सा",
        "नफरत",
    }

    fear_terms = {
        "afraid",
        "fear",
        "panic",
        "scared",
        "threat",
        "unsafe",
        "warning",
        "worry",
        "anxious",
        "frightened",
        "डर",
        "खतरा",
    }

    hate_terms = {
        "contaminate",
        "eliminate",
        "enemy",
        "filth",
        "hate",
        "impure",
        "invader",
        "outsider",
        "purge",
        "subhuman",
        "vermin",
        "घृणा",
        "दुश्मन",
        "नफरत",
    }

    def __init__(self, preprocessor: NLPPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or NLPPreprocessor(use_lemmatization=False)

    def _score(self, tokens: List[str], lexicon: set[str]) -> float:
        if not tokens:
            return 0.0
        matches = sum(1 for token in tokens if token in lexicon)
        return round(min(100.0, (matches / max(len(tokens), 1)) * 220.0), 2)

    def analyze(self, text: str) -> Dict[str, object]:
        tokens = self.preprocessor.preprocess(text).tokens
        anger = self._score(tokens, self.anger_terms)
        fear = self._score(tokens, self.fear_terms)
        hate = self._score(tokens, self.hate_terms)

        scores = {"anger": anger, "fear": fear, "hate": hate}
        dominant = max(scores, key=scores.get) if scores else "neutral"
        if scores[dominant] < 12:
            dominant = "neutral"

        return {
            "label": dominant,
            "scores": scores,
        }


DEFAULT_PREPROCESSOR = NLPPreprocessor()


def normalize_text(text: str) -> str:
    return DEFAULT_PREPROCESSOR.preprocess(text).normalized_text


def vectorizer_preprocessor(text: str) -> str:
    return normalize_text(text)


def vectorizer_tokenizer(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split()


def truncate_preview(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

