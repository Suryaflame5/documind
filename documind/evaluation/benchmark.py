"""
DocuMind Comprehensive Benchmark & Model Evaluation Engine
Trains and benchmarks Baseline (TF-IDF + LR) vs Deep Learning (PyTorch Transformer).
Measures Accuracy, Precision, Recall, Macro-F1, Latency, and Confusion Matrices on unseen test data.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from documind.core.config import settings
from documind.models.dataset_generator import EnterpriseDatasetGenerator
from documind.models.classifier_baseline import BaselineDocumentClassifier
from documind.models.classifier_dl import DeepLearningDocumentClassifier


class ModelBenchmarkRunner:
    def __init__(self, samples_per_class: int = 75):
        self.generator = EnterpriseDatasetGenerator(seed=42)
        self.samples_per_class = samples_per_class
        self.categories = settings.CATEGORIES

    def run_full_benchmark(self) -> Dict[str, Any]:
        print(f"[*] Generating dataset: {self.samples_per_class} samples per class across {len(self.categories)} categories...")
        train_data, test_data = self.generator.generate_dataset(samples_per_class=self.samples_per_class)
        print(f"[*] Dataset split: {len(train_data)} train samples, {len(test_data)} test samples.")

        train_texts = [d["text"] for d in train_data]
        train_labels = [d["category"] for d in train_data]
        test_texts = [d["text"] for d in test_data]
        test_labels = [d["category"] for d in test_data]

        # -------------------------------------------------------------
        # 1. Baseline Model Training & Evaluation
        # -------------------------------------------------------------
        print("\n--- Training Baseline Model (TF-IDF + Calibrated Logistic Regression) ---")
        t0 = time.time()
        baseline = BaselineDocumentClassifier()
        baseline.train(train_texts, train_labels)
        baseline_train_time = round(time.time() - t0, 3)
        baseline.save()
        print(f"[*] Baseline trained in {baseline_train_time}s and saved to {settings.BASELINE_MODEL_PATH}")

        # Baseline Test Inference
        baseline_preds = []
        baseline_latencies = []
        for text in test_texts:
            t_start = time.time()
            res = baseline.predict(text)
            latency = (time.time() - t_start) * 1000.0
            baseline_latencies.append(latency)
            baseline_preds.append(res.predicted_category)

        baseline_acc = float(accuracy_score(test_labels, baseline_preds))
        b_prec, b_rec, b_f1, _ = precision_recall_fscore_support(
            test_labels, baseline_preds, average="macro", zero_division=0
        )
        b_class_p, b_class_r, b_class_f1, b_support = precision_recall_fscore_support(
            test_labels, baseline_preds, labels=self.categories, zero_division=0
        )
        b_cm = confusion_matrix(test_labels, baseline_preds, labels=self.categories).tolist()

        baseline_results = {
            "model_name": "Baseline (TF-IDF + Logistic Regression)",
            "accuracy": round(baseline_acc, 4),
            "macro_precision": round(float(b_prec), 4),
            "macro_recall": round(float(b_rec), 4),
            "macro_f1": round(float(b_f1), 4),
            "mean_latency_ms": round(float(np.mean(baseline_latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(baseline_latencies, 95)), 2),
            "training_time_s": baseline_train_time,
            "confusion_matrix": {
                "labels": self.categories,
                "matrix": b_cm
            },
            "per_class_metrics": {
                self.categories[i]: {
                    "precision": round(float(b_class_p[i]), 4),
                    "recall": round(float(b_class_r[i]), 4),
                    "f1_score": round(float(b_class_f1[i]), 4),
                    "support": int(b_support[i])
                }
                for i in range(len(self.categories))
            }
        }

        # -------------------------------------------------------------
        # 2. Deep Learning Model Training & Evaluation
        # -------------------------------------------------------------
        print("\n--- Training Deep Learning Model (PyTorch Multi-Head Self-Attention Transformer) ---")
        t0 = time.time()
        dl_model = DeepLearningDocumentClassifier()
        dl_model.train(train_texts, train_labels, epochs=10, batch_size=16, lr=1e-3)
        dl_train_time = round(time.time() - t0, 3)
        dl_model.save()
        print(f"[*] Deep Learning model trained in {dl_train_time}s and saved to {settings.DL_MODEL_PATH}")

        # DL Test Inference
        dl_preds = []
        dl_latencies = []
        for text in test_texts:
            t_start = time.time()
            res = dl_model.predict(text)
            latency = (time.time() - t_start) * 1000.0
            dl_latencies.append(latency)
            dl_preds.append(res.predicted_category)

        dl_acc = float(accuracy_score(test_labels, dl_preds))
        d_prec, d_rec, d_f1, _ = precision_recall_fscore_support(
            test_labels, dl_preds, average="macro", zero_division=0
        )
        d_class_p, d_class_r, d_class_f1, d_support = precision_recall_fscore_support(
            test_labels, dl_preds, labels=self.categories, zero_division=0
        )
        d_cm = confusion_matrix(test_labels, dl_preds, labels=self.categories).tolist()

        dl_results = {
            "model_name": "Deep Learning (PyTorch Attention Transformer)",
            "accuracy": round(dl_acc, 4),
            "macro_precision": round(float(d_prec), 4),
            "macro_recall": round(float(d_rec), 4),
            "macro_f1": round(float(d_f1), 4),
            "mean_latency_ms": round(float(np.mean(dl_latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(dl_latencies, 95)), 2),
            "training_time_s": dl_train_time,
            "confusion_matrix": {
                "labels": self.categories,
                "matrix": d_cm
            },
            "per_class_metrics": {
                self.categories[i]: {
                    "precision": round(float(d_class_p[i]), 4),
                    "recall": round(float(d_class_r[i]), 4),
                    "f1_score": round(float(d_class_f1[i]), 4),
                    "support": int(d_support[i])
                }
                for i in range(len(self.categories))
            }
        }

        # -------------------------------------------------------------
        # 3. Model Comparison Summary
        # -------------------------------------------------------------
        benchmark_report = {
            "dataset_info": {
                "categories": self.categories,
                "total_samples": len(train_data) + len(test_data),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "test_ratio": 0.20
            },
            "models": {
                "baseline": baseline_results,
                "deep_learning": dl_results
            },
            "summary": {
                "accuracy_delta": round(dl_acc - baseline_acc, 4),
                "f1_delta": round(float(d_f1 - b_f1), 4),
                "winner": "deep_learning" if dl_acc >= baseline_acc else "baseline"
            }
        }

        with open(settings.BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2)

        print(f"\n[+] Benchmark completed successfully! Report saved to {settings.BENCHMARK_REPORT_PATH}")
        return benchmark_report


if __name__ == "__main__":
    runner = ModelBenchmarkRunner(samples_per_class=60)
    res = runner.run_full_benchmark()
    print(json.dumps(res["summary"], indent=2))
