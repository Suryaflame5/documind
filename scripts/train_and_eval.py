"""
End-to-End Training and Evaluation Pipeline
Trains Baseline and Deep Learning models, computes comprehensive classification and extraction
benchmarks on unseen test data, and saves model checkpoints and evaluation reports.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from documind.core.config import settings
from documind.evaluation.benchmark import ModelBenchmarkRunner
from documind.evaluation.field_evaluator import FieldExtractionEvaluator


def main():
    print("=" * 80)
    print("  DOCUMIND: ENTERPRISE DOCUMENT INTELLIGENCE MODEL TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Run Model Classification Benchmark (Baseline vs Deep Learning)
    print("\n>>> STAGE 1: Document Classification Benchmark (Baseline vs PyTorch Transformer)")
    benchmark_runner = ModelBenchmarkRunner(samples_per_class=60)
    bench_results = benchmark_runner.run_full_benchmark()

    # 2. Run Field Extraction Evaluation
    print("\n>>> STAGE 2: Field Extraction Evaluation (Exact Match & Field F1)")
    field_evaluator = FieldExtractionEvaluator()
    field_results = field_evaluator.evaluate_extraction(num_samples=50)

    # 3. Print Comprehensive Report Summary
    print("\n" + "=" * 80)
    print("                           EVALUATION SUMMARY REPORT")
    print("=" * 80)

    base = bench_results["models"]["baseline"]
    dl = bench_results["models"]["deep_learning"]

    print("\n[DOCUMENT CLASSIFICATION METRICS (Unseen Held-out Test Set)]")
    print(f"{'Metric':<25} {'Baseline (TF-IDF + LR)':<25} {'Deep Learning (PyTorch)':<25}")
    print("-" * 75)
    print(f"{'Accuracy':<25} {base['accuracy']*100:.2f}%{'':<18} {dl['accuracy']*100:.2f}%")
    print(f"{'Macro Precision':<25} {base['macro_precision']*100:.2f}%{'':<18} {dl['macro_precision']*100:.2f}%")
    print(f"{'Macro Recall':<25} {base['macro_recall']*100:.2f}%{'':<18} {dl['macro_recall']*100:.2f}%")
    print(f"{'Macro F1-Score':<25} {base['macro_f1']*100:.2f}%{'':<18} {dl['macro_f1']*100:.2f}%")
    print(f"{'Mean Latency':<25} {base['mean_latency_ms']:.2f} ms{'':<16} {dl['mean_latency_ms']:.2f} ms")
    print(f"{'P95 Latency':<25} {base['p95_latency_ms']:.2f} ms{'':<16} {dl['p95_latency_ms']:.2f} ms")

    print("\n[FIELD EXTRACTION METRICS]")
    print(f"{'Field':<20} {'Exact Match (EM)':<20} {'Token F1-Score':<20}")
    print("-" * 60)
    for field_name, metrics in field_results["field_metrics"].items():
        em = metrics["exact_match_accuracy"] * 100
        f1 = metrics["mean_token_f1"] * 100
        print(f"{field_name:<20} {em:.2f}%{'':<13} {f1:.2f}%")

    print("-" * 60)
    print(f"Mathematical Consistency Pass Rate: {field_results['mathematical_consistency_pass_rate']*100:.2f}%")
    print(f"Mean Extraction Latency:           {field_results['mean_extraction_latency_ms']:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
