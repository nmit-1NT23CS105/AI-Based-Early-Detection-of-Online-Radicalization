from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from preprocessing import normalize_text, vectorizer_preprocessor, vectorizer_tokenizer


LABEL_ORDER = ["normal", "suspicious", "radical"]


class LexiconSignalTransformer(BaseEstimator, TransformerMixin):
    suspicious_terms = {
        "attack",
        "betray",
        "coded",
        "confrontation",
        "corrupted",
        "enemy",
        "fail",
        "fear",
        "group",
        "hidden",
        "identity",
        "insider",
        "loyal",
        "outsider",
        "prepare",
        "push",
        "threat",
        "truth",
        "us",
        "versus",
        "चुप",
        "डर",
        "खतरा",
        "कमजोरी",
        "टकराव",
        "दुश्मन",
    }

    radical_terms = {
        "attack",
        "crush",
        "destroy",
        "eliminate",
        "fear",
        "force",
        "hurt",
        "kill",
        "pain",
        "punish",
        "purify",
        "retaliation",
        "revenge",
        "strike",
        "violent",
        "बल",
        "बदला",
        "हिंसक",
        "कुचलना",
        "चोट",
        "सख्त",
    }

    normal_terms = {
        "analysis",
        "awareness",
        "calm",
        "class",
        "community",
        "dialogue",
        "discussion",
        "education",
        "empathy",
        "ethics",
        "lab",
        "library",
        "meeting",
        "peaceful",
        "project",
        "research",
        "responsible",
        "seminar",
        "team",
        "workshop",
        "awareness",
        "जिम्मेदार",
        "संवाद",
        "शांत",
        "सेमिनार",
    }

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        rows = []
        for text in x:
            tokens = normalize_text(text).split()
            length = max(len(tokens), 1)
            suspicious_hits = sum(token in self.suspicious_terms for token in tokens)
            radical_hits = sum(token in self.radical_terms for token in tokens)
            normal_hits = sum(token in self.normal_terms for token in tokens)
            rows.append(
                [
                    suspicious_hits,
                    radical_hits,
                    normal_hits,
                    suspicious_hits / length,
                    radical_hits / length,
                    normal_hits / length,
                ]
            )
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "lexicon__suspicious_hits",
                "lexicon__radical_hits",
                "lexicon__normal_hits",
                "lexicon__suspicious_ratio",
                "lexicon__radical_ratio",
                "lexicon__normal_ratio",
            ]
        )


class TransformerEmbeddingExtractor:
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased") -> None:
        self.model_name = model_name
        self.available = False
        self.status = "Transformer backend is disabled."
        self.tokenizer = None
        self.model = None

        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.available = True
            self.status = f"Transformer backend ready with {model_name}."
        except Exception as exc:
            self.status = (
                f"Transformer backend unavailable ({exc.__class__.__name__}: {exc}). "
                "The project will run with the TF-IDF classifier unless PyTorch and model "
                "weights are available."
            )

    def describe(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "available": self.available,
            "status": self.status,
        }


class RadicalizationModelManager:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "radicalization_model.joblib"
        self.metrics_path = self.model_dir / "metrics.json"
        self.metadata_path = self.model_dir / "metadata.json"
        self.default_dataset = self.data_dir / "combined_training_dataset.csv"
        self.transformer = TransformerEmbeddingExtractor()

    def build_pipeline(self) -> Pipeline:
        word_vectorizer = TfidfVectorizer(
            preprocessor=vectorizer_preprocessor,
            tokenizer=vectorizer_tokenizer,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=7000,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            preprocessor=vectorizer_preprocessor,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=5000,
            sublinear_tf=True,
        )
        features = FeatureUnion(
            transformer_list=[
                ("word_tfidf", word_vectorizer),
                ("char_tfidf", char_vectorizer),
                ("lexicon_signals", LexiconSignalTransformer()),
            ]
        )
        classifier = LogisticRegression(
            max_iter=3500,
            class_weight="balanced",
            solver="lbfgs",
        )
        return Pipeline(
            steps=[
                ("features", features),
                ("classifier", classifier),
            ]
        )

    def _load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        frame = pd.read_csv(dataset_path)
        required = {"text", "label"}
        if not required.issubset(frame.columns):
            raise ValueError("Dataset must contain 'text' and 'label' columns.")
        frame = frame.dropna(subset=["text", "label"]).copy()
        frame["text"] = frame["text"].astype(str).str.strip()
        frame["label"] = frame["label"].astype(str).str.strip().str.lower()
        frame = frame[frame["label"].isin(LABEL_ORDER)]
        frame = frame[frame["text"].str.len() > 3]
        frame = frame.drop_duplicates(subset=["text"])
        return frame.reset_index(drop=True)

    def _top_terms(self, pipeline: Pipeline) -> Dict[str, list[Dict[str, float]]]:
        classifier = pipeline.named_steps["classifier"]
        features = pipeline.named_steps["features"].get_feature_names_out()
        insights: Dict[str, list[Dict[str, float]]] = {}
        for class_index, label in enumerate(classifier.classes_):
            weights = classifier.coef_[class_index]
            top_indices = weights.argsort()[-12:][::-1]
            insights[label] = [
                {"term": str(features[index]).split("__")[-1], "weight": round(float(weights[index]), 4)}
                for index in top_indices
            ]
        return insights

    def _build_metrics(
        self,
        y_true: pd.Series,
        y_pred: list[str],
        pipeline: Pipeline,
        dataset_size: int,
    ) -> Dict[str, Any]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            average="weighted",
            zero_division=0,
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": matrix.tolist(),
            "labels": LABEL_ORDER,
            "classification_report": report,
            "dataset_size": dataset_size,
            "top_terms": self._top_terms(pipeline),
        }

    def train_from_csv(self, dataset_path: str | Path | None = None) -> Dict[str, Any]:
        dataset_path = Path(dataset_path or self.default_dataset)
        frame = self._load_dataset(dataset_path)
        if frame.shape[0] < 24:
            raise ValueError("The dataset must contain at least 24 records for a stable split.")

        x_train, x_test, y_train, y_test = train_test_split(
            frame["text"],
            frame["label"],
            test_size=0.25,
            random_state=42,
            stratify=frame["label"],
        )

        pipeline = self.build_pipeline()
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = self._build_metrics(y_test, list(predictions), pipeline, dataset_size=int(frame.shape[0]))

        label_distribution = frame["label"].value_counts().sort_index().to_dict()
        metadata = {
            "model_type": "Hybrid TF-IDF + Logistic Regression",
            "feature_extractors": [
                "word-level TF-IDF",
                "character-level TF-IDF",
                "transformer-ready integration hook",
            ],
            "labels": LABEL_ORDER,
            "dataset_path": str(dataset_path),
            "dataset_size": int(frame.shape[0]),
            "label_distribution": label_distribution,
            "transformer_backend": self.transformer.describe(),
        }

        joblib.dump(pipeline, self.model_path)
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"metrics": metrics, "metadata": metadata}

    def load_artifacts(self) -> Tuple[Pipeline | None, Dict[str, Any], Dict[str, Any]]:
        pipeline = joblib.load(self.model_path) if self.model_path.exists() else None
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8")) if self.metadata_path.exists() else {}
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8")) if self.metrics_path.exists() else {}
        return pipeline, metadata, metrics

    def ensure_model(self) -> Dict[str, Any]:
        if self.model_path.exists() and self.metrics_path.exists() and self.metadata_path.exists():
            _, metadata, metrics = self.load_artifacts()
            return {"metrics": metrics, "metadata": metadata}
        return self.train_from_csv(self.default_dataset)
