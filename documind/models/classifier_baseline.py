"""
Baseline Document Classifier
TF-IDF Feature Extraction with Calibrated Logistic Regression.
Provides fast, interpretable, low-latency document classification with calibrated probabilities.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from documind.core.config import settings
from documind.extraction.schema import DocumentClassificationResult


class BaselineDocumentClassifier:
    def __init__(self):
        self.categories = settings.CATEGORIES
        self.pipeline: Pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=6000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b"
            )),
            ("clf", LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ])
        self.is_trained: bool = False

    def train(self, texts: List[str], labels: List[str]):
        """
        Trains the TF-IDF + Logistic Regression pipeline.
        """
        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def predict(self, text: str) -> DocumentClassificationResult:
        """
        Predicts category and class probabilities for a single document text.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained or loaded yet.")

        probs = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        
        prob_dict = {cls_name: float(round(probs[idx], 4)) for idx, cls_name in enumerate(classes)}
        best_idx = int(np.argmax(probs))
        best_category = classes[best_idx]
        best_confidence = float(round(probs[best_idx], 4))

        return DocumentClassificationResult(
            predicted_category=best_category,
            confidence=best_confidence,
            probabilities=prob_dict,
            model_used="baseline_tfidf"
        )

    def save(self, model_path: Path = settings.BASELINE_MODEL_PATH):
        """
        Serializes the trained pipeline to disk.
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_path)

    def load(self, model_path: Path = settings.BASELINE_MODEL_PATH) -> bool:
        """
        Loads the serialized pipeline from disk.
        """
        if not model_path.exists():
            return False
        self.pipeline = joblib.load(model_path)
        self.is_trained = True
        return True
