from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List

from lime.lime_text import LimeTextExplainer

from model import LABEL_ORDER, RadicalizationModelManager
from preprocessing import EmotionAnalyzer, NLPPreprocessor


DISPLAY_LABELS = {
    "normal": "Normal",
    "suspicious": "Suspicious",
    "radical": "Radical",
}

LABEL_COLORS = {
    "normal": "#2BB673",
    "suspicious": "#F59E0B",
    "radical": "#E34F4F",
}


class PredictionService:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.preprocessor = NLPPreprocessor()
        self.emotion_analyzer = EmotionAnalyzer(self.preprocessor)
        self.model_manager = RadicalizationModelManager(self.base_dir)
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.explainer = LimeTextExplainer(class_names=[DISPLAY_LABELS[label] for label in LABEL_ORDER])
        self.reload()

    def reload(self) -> None:
        self.model_manager.ensure_model()
        self.pipeline, self.metadata, self.metrics = self.model_manager.load_artifacts()
        if self.pipeline is None:
            raise RuntimeError("Model artifacts are missing. Run train.py to generate them.")

    def _classifier_fn(self, texts: List[str]):
        return self.pipeline.predict_proba(texts)

    def _build_explanation(self, text: str, predicted_label: str, top_n: int = 8) -> Dict[str, Any]:
        try:
            predicted_index = LABEL_ORDER.index(predicted_label)
            explanation = self.explainer.explain_instance(
                text,
                self._classifier_fn,
                labels=[predicted_index],
                num_features=top_n,
                num_samples=120,
            )
            weighted_terms = explanation.as_list(label=predicted_index)
        except Exception:
            weighted_terms = self._fallback_explanation(text, predicted_label, top_n=top_n)

        top_features = [
            {
                "term": term,
                "weight": round(float(weight), 4),
                "direction": "supports" if weight >= 0 else "reduces",
            }
            for term, weight in weighted_terms
        ]

        return {
            "top_features": top_features,
            "highlighted_html": self._render_highlighted_text(text, top_features),
        }

    def _fallback_explanation(self, text: str, predicted_label: str, top_n: int = 8):
        classifier = self.pipeline.named_steps["classifier"]
        features = self.pipeline.named_steps["features"]
        feature_names = features.get_feature_names_out()
        feature_vector = features.transform([text])
        label_index = list(classifier.classes_).index(predicted_label)
        coefficients = classifier.coef_[label_index]
        weighted_vector = feature_vector.multiply(coefficients)
        ranked = weighted_vector.toarray()[0].argsort()[-top_n:][::-1]
        results = []
        for index in ranked:
            value = float(weighted_vector[0, index])
            if value <= 0:
                continue
            term = str(feature_names[index]).split("__")[-1]
            results.append((term, value))
        return results[:top_n]

    def _render_highlighted_text(self, text: str, features: List[Dict[str, Any]]) -> str:
        weights = {}
        for item in features:
            key = item["term"].lower().strip()
            if key:
                weights[key] = item["weight"]

        chunks = re.findall(r"[\u0900-\u097FA-Za-z']+|\s+|[^\w\s]", text)
        rendered = []
        for chunk in chunks:
            normalized = chunk.lower()
            if normalized.strip() and normalized in weights:
                strength = min(0.92, 0.2 + abs(weights[normalized]))
                background = "rgba(227,79,79,{})".format(strength) if weights[normalized] > 0 else "rgba(43,182,115,{})".format(strength)
                rendered.append(
                    f"<span class='token-highlight' style='background:{background}'>{html.escape(chunk)}</span>"
                )
            else:
                rendered.append(html.escape(chunk))
        return "".join(rendered)

    def _risk_assessment(
        self,
        predicted_label: str,
        probabilities: Dict[str, float],
        emotion: Dict[str, Any],
    ) -> Dict[str, Any]:
        radical_component = probabilities.get("radical", 0.0) * 78
        suspicious_component = probabilities.get("suspicious", 0.0) * 46
        anger_component = emotion["scores"]["anger"] * 0.16
        hate_component = emotion["scores"]["hate"] * 0.2
        fear_component = emotion["scores"]["fear"] * 0.08
        bias = 8 if predicted_label == "radical" else 3 if predicted_label == "suspicious" else 0
        score = min(100, round(radical_component + suspicious_component + anger_component + hate_component + fear_component + bias))

        if score >= 70:
            level = "High Risk"
        elif score >= 35:
            level = "Medium Risk"
        else:
            level = "Low Risk"

        return {
            "score": score,
            "level": level,
            "alert": level == "High Risk",
        }

    def predict_one(self, text: str) -> Dict[str, Any]:
        processed = self.preprocessor.preprocess(text)
        probabilities_array = self.pipeline.predict_proba([processed.original_text])[0]
        class_names = list(self.pipeline.named_steps["classifier"].classes_)
        probability_map = {label: round(float(prob), 4) for label, prob in zip(class_names, probabilities_array)}
        predicted_label = max(probability_map, key=probability_map.get)
        confidence = round(probability_map[predicted_label], 4)
        emotion = self.emotion_analyzer.analyze(processed.original_text)
        risk = self._risk_assessment(predicted_label, probability_map, emotion)
        explanation = self._build_explanation(processed.original_text, predicted_label)

        return {
            "original_text": processed.original_text,
            "anonymized_text": processed.anonymized_text,
            "input_hash": processed.input_hash,
            "language": processed.language,
            "prediction": {
                "label": DISPLAY_LABELS[predicted_label],
                "label_key": predicted_label,
                "color": LABEL_COLORS[predicted_label],
                "confidence": confidence,
                "probabilities": probability_map,
            },
            "emotion": emotion,
            "risk": risk,
            "explanation": explanation,
            "model_info": self.metadata,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict_one(text) for text in texts]

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def get_model_metadata(self) -> Dict[str, Any]:
        return self.metadata
