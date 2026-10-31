"""
Field Extraction Evaluator
Evaluates information extraction accuracy on unseen held-out enterprise documents:
Exact Match (EM) accuracy, field-level token F1, and mathematical consistency pass rates.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from documind.core.config import settings
from documind.models.dataset_generator import EnterpriseDatasetGenerator
from documind.extraction.extractors import DocumentFieldExtractor
from documind.extraction.validator import DocumentValidator
from documind.models.classifier_baseline import BaselineDocumentClassifier


def compute_token_f1(pred_str: str, truth_str: str) -> float:
    """
    Computes token-level precision, recall, and F1 between predicted string and ground-truth.
    """
    pred_tokens = str(pred_str).lower().replace(",", "").split()
    truth_tokens = str(truth_str).lower().replace(",", "").split()
    
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(tok), truth_tokens.count(tok)) for tok in common)
    
    if num_same == 0:
        return 0.0
        
    prec = 1.0 * num_same / len(pred_tokens)
    rec = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


class FieldExtractionEvaluator:
    def __init__(self):
        self.extractor = DocumentFieldExtractor()
        self.validator = DocumentValidator()
        self.generator = EnterpriseDatasetGenerator(seed=123)

    def evaluate_extraction(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Runs field extraction over generated held-out test documents and computes Exact Match and F1.
        """
        _, test_samples = self.generator.generate_dataset(samples_per_class=num_samples // 4 + 5)
        test_samples = test_samples[:num_samples]

        field_names = ["document_id", "date", "total_amount", "currency", "vendor_name", "customer_name"]
        exact_matches = {f: 0 for f in field_names}
        f1_scores = {f: [] for f in field_names}
        extraction_latencies = []
        math_valid_count = 0
        invoice_receipt_count = 0
        failure_cases = []

        for sample in test_samples:
            raw_text = sample["text"]
            meta = sample.get("metadata", {})
            cat = sample["category"]

            t0 = time.time()
            extracted_fields, line_items = self.extractor.extract_all(raw_text, spans=[])
            extraction_latencies.append((time.time() - t0) * 1000.0)

            # Evaluate each target field
            for f in field_names:
                gt_val = str(meta.get(f, "")).strip()
                pred_obj = extracted_fields.get(f)
                pred_val = str(pred_obj.value).strip() if pred_obj else ""

                if f in ["total_amount", "subtotal", "tax"]:
                    try:
                        gt_num = float(gt_val)
                        pred_num = float(pred_val)
                        if abs(gt_num - pred_num) < 0.02:
                            exact_matches[f] += 1
                            f1_scores[f].append(1.0)
                        else:
                            f1_scores[f].append(0.0)
                            if len(failure_cases) < 5 and gt_num > 0:
                                failure_cases.append({
                                    "field": f,
                                    "expected": f"{gt_num:.2f}",
                                    "predicted": f"{pred_num:.2f}",
                                    "category": cat
                                })
                        continue
                    except ValueError:
                        pass

                # Exact match
                if gt_val and pred_val and (gt_val.lower() == pred_val.lower() or gt_val == pred_val):
                    exact_matches[f] += 1
                else:
                    if len(failure_cases) < 5 and gt_val != "0.00" and gt_val:
                        failure_cases.append({
                            "field": f,
                            "expected": gt_val,
                            "predicted": pred_val,
                            "category": cat
                        })

                # Token F1
                score_f1 = compute_token_f1(pred_val, gt_val)
                f1_scores[f].append(score_f1)

            # Mathematical validation rate for invoices & receipts
            if cat in ["invoice", "receipt"]:
                invoice_receipt_count += 1
                sub = float(extracted_fields["subtotal"].value)
                tx = float(extracted_fields["tax"].value)
                tot = float(extracted_fields["total_amount"].value)
                if abs((sub + tx) - tot) <= 0.05:
                    math_valid_count += 1

        total_tested = len(test_samples)
        field_metrics = {}
        for f in field_names:
            em_rate = round(exact_matches[f] / total_tested, 4)
            mean_f1 = round(float(np.mean(f1_scores[f])), 4)
            field_metrics[f] = {
                "exact_match_accuracy": em_rate,
                "mean_token_f1": mean_f1
            }

        math_pass_rate = round(math_valid_count / max(1, invoice_receipt_count), 4)

        report = {
            "total_documents_tested": total_tested,
            "field_metrics": field_metrics,
            "mean_extraction_latency_ms": round(float(np.mean(extraction_latencies)), 2),
            "p95_extraction_latency_ms": round(float(np.percentile(extraction_latencies, 95)), 2),
            "mathematical_consistency_pass_rate": math_pass_rate,
            "failure_case_samples": failure_cases
        }

        # Save to artifacts
        with open(settings.EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    evaluator = FieldExtractionEvaluator()
    res = evaluator.evaluate_extraction(num_samples=40)
    print(json.dumps(res, indent=2))
