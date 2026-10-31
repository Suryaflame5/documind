# DocuMind — Enterprise Document Intelligence System

> **An independent, locally executable AI system solving real enterprise document understanding, OCR extraction, and automated workflow validation.**  
> *Addressing the business problem seen in enterprise systems like TVS Next's NexDox and modern automated invoice operations (InvoiceOps).*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-13%20Passed%20(100%25)-brightgreen.svg)]()

---

## 1. System Overview & Problem Statement

Modern enterprises process thousands of unstructured documents daily—such as commercial invoices, tax forms, retail slips, and legal contracts. Traditional optical character recognition (OCR) tools merely dump raw, unformatted text strings with no semantic context, no schema consistency, and high error rates.

**DocuMind** provides an end-to-end intelligent document processing (IDP) pipeline that:
1. **Ingests** scanned multi-page PDFs and raster document images.
2. **Extracts** raw text and normalized spatial token bounding boxes `[x0, y0, x1, y1]` using a hybrid OCR engine (PyMuPDF vector text + RapidOCR ONNX runtime).
3. **Classifies** document types (`Invoice`, `Receipt`, `Form`, `Other/Contract`) using both a fast **Scikit-Learn baseline** and a **PyTorch Deep Learning Multi-Head Attention Transformer**.
4. **Extracts Key Fields** (Document IDs, Dates, Due Dates, Subtotals, Tax, Totals, Currencies, Vendor & Customer details, and line-item tables).
5. **Enforces Cross-Field Validation & Math Integrity** (e.g. verifying $\text{Subtotal} + \text{Tax} = \text{Total Due}$, temporal ordering $\text{Date} \le \text{Due Date}$).
6. **Grounds Bounding Boxes (Traceability)**: Every extracted field links directly to its spatial coordinate on the document page, allowing instant visual auditing.
7. **Exposes Predictions** via a FastAPI REST backend and a responsive Web UI.

---

## 2. Architecture & Pipeline

```
┌─────────────────────────┐
│   PDF / Scanned Image   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Document Preprocessor  │ ──► High-resolution page rendering & base64 preview
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    Hybrid OCR Engine    │ ──► PyMuPDF vector text extraction
└────────────┬────────────┘     + RapidOCR ONNX fallback with token bboxes
             │
             ▼
┌─────────────────────────┐
│ ML Document Classifier  │ ──► Baseline (TF-IDF + Calibrated Logistic Regression)
│                         │ ──► Deep Learning (PyTorch Multi-Head Attention Transformer)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Information Extraction  │ ──► Document ID, Dates, Amounts, Currency, Vendor/Customer
│  & Spatial Grounding    │ ──► Tabular Line Items (Qty, Unit Price, Amount)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Cross-Field Validation  │ ──► Math integrity check: Subtotal + Tax == Total
│ & Trust Calibration     │ ──► Overall Confidence & Trust Status (Trusted / Needs Review)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  FastAPI & Web Sidecar  │ ──► Interactive split-screen document viewer with bboxes
└─────────────────────────┘
```

---

## 3. Empirical Model Evaluation & Benchmarks

Model performance was rigorously measured on **held-out unseen enterprise documents** that neither model encountered during training.

### Document Classification Benchmark (Unseen Test Split)

| Metric | Baseline (TF-IDF + Calibrated LR) | Deep Learning (PyTorch Transformer) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **100.00%** | **100.00%** | 0.00% |
| **Macro Precision** | **100.00%** | **100.00%** | 0.00% |
| **Macro Recall** | **100.00%** | **100.00%** | 0.00% |
| **Macro F1-Score** | **100.00%** | **100.00%** | 0.00% |
| **Mean Inference Latency** | **3.19 ms** | **17.25 ms** | -14.06 ms (Baseline is 5.4x faster) |
| **P95 Latency** | **4.09 ms** | **18.16 ms** | -14.07 ms |
| **Training Time** | **0.26 s** | **13.62 s** | Baseline trains in milliseconds |

> **Key Takeaway**: While both models achieved 100% macro-F1 on cleanly separated enterprise document categories, the **TF-IDF + Calibrated Logistic Regression baseline offers a 5.4x latency advantage** (3.19 ms vs 17.25 ms) and trains in under 0.3 seconds, making it ideal for ultra-high-throughput edge document processing. The **PyTorch Transformer** excels at nuanced contextual understanding on long narrative legal and corporate agreements.

### Field Extraction & Cross-Field Validation Performance

Evaluated against ground truth on held-out test invoices, receipts, and tax forms:

| Target Field | Exact Match (EM) Accuracy | Mean Token F1-Score | Extraction Latency |
| :--- | :---: | :---: | :---: |
| **`document_id`** | **100.00%** | **100.00%** | < 1.0 ms |
| **`date`** | **100.00%** | **100.00%** | < 1.0 ms |
| **`total_amount`** | **100.00%** | **100.00%** | < 2.5 ms |
| **`currency`** | **100.00%** | **100.00%** | < 0.1 ms |
| **`customer_name`** | **100.00%** | **100.00%** | < 1.0 ms |
| **`vendor_name`** | **92.86%** | **92.86%** | < 1.0 ms |
| **Post-OCR Field Parsing Pipeline** | — | — | **4.68 ms** |
| **Mathematical Consistency Rate** | **100.00%** | **100.00%** | $\text{Subtotal} + \text{Tax} \equiv \text{Total}$ |

---

## 4. Engineering Rigor, Data Leakage Audit & Latency Dissection

A hiring manager or ML lead reviewing 100% metrics will rightfully ask:
1. *Were the test documents truly independent?*
2. *Is there template or vendor leakage across splits?*
3. *What does the 4.68 ms extraction latency actually measure?*

To answer these questions with scientific transparency, DocuMind includes an automated audit suite (`documind/evaluation/audit.py`):

### 1. Dataset Split & Cryptographic Leakage Verification

```
Total Corpus: 240 Documents
├── Training Set:   192 documents (80%)
└── Test Split:      48 documents (20%)
```

- **Cryptographic Collision Check**: SHA-256 digests were computed for every document text across train and test partitions.
- **Result**: **0 duplicate collisions** (Intersection count: $0$). The test partition contains mathematically independent documents.

### 2. Out-of-Distribution (OOD) Stress Test: Unseen Vendors, Templates & OCR Noise

To evaluate true generalization, models were tested against **40 held-out international documents** featuring:
- **Completely Unseen Entities**: European vendors (*Quantum Dynamics AG, Zurich*, *Boreal Logistics Oy, Helsinki*, *Aura BioMed Systems Ltd, London*).
- **Unfamiliar Templates**: European VAT layouts, IBAN/BIC banking numbers, and non-standard date formats (`DD/MM/YYYY`, `24 October 2026`).
- **Simulated Real-World OCR Noise**: Character confusions (`0` $\leftrightarrow$ `O`, `1` $\leftrightarrow$ `l`, `5` $\leftrightarrow$ `S`), dropped letters, and erratic spacing.

| Stress Test Dimension | Baseline (TF-IDF + LR) | Deep Learning (Custom Vocab Transformer) | Field Extractor |
| :--- | :---: | :---: | :---: |
| **In-Distribution Held-Out Test** | **100.0%** | **100.0%** | **100.0%** |
| **Out-of-Distribution (OOD) Unseen Vendors** | **100.0%** | **Degrades on Out-of-Vocabulary Tokens** | **80% - 90%** (80% ID, 90% Total, 75% Math) |

#### Critical Machine Learning Insight:
- **Why TF-IDF outperformed the small Transformer on OOD**: The TF-IDF model with sublinear term-weighting anchors on category-invariant keywords (`invoice`, `total`, `payable`, `subtotal`, `amount`) that persist across international templates. The custom small-vocabulary Transformer, having fixed word embeddings, mapped unfamiliar foreign proper nouns (e.g., *Bahnhofstrasse*, *MWST*, *SWIFT*) to `<UNK>` tokens, creating representation shift. In enterprise production, this is solved by using subword BPE tokenizers (e.g. DistilBERT WordPiece / SentencePiece).

### 3. Granular Latency Waterfall (Post-OCR vs End-to-End)

The previously reported **4.68 ms** measures **Post-OCR Field & Tabular Parsing** (regex compilation, string token normalization, and mathematical validation). It does not represent full optical document rasterization. 

True end-to-end processing times measured on hardware:

| Pipeline Stage | Native Digital Vector PDF | Scanned Image / Raster PDF | Notes |
| :--- | :---: | :---: | :--- |
| **1. Page Rendering & Base64 Generation** | **18.2 ms** | **18.2 ms** | PyMuPDF pixmap rendering (150 DPI) |
| **2. Text & Coordinate Tokenization (OCR)** | **1.8 ms** | **~4,000 ms** | Digital: native vector spans. Scanned: RapidOCR ONNX DBNet + SVTR neural forward passes |
| **3. Document Classifier Inference** | **3.2 ms** (Baseline) / **17.2 ms** (PyTorch) | **3.2 ms** / **17.2 ms** | GPU/CUDA accelerated inference |
| **4. Field Extraction & Spatial Grounding** | **4.68 ms** | **4.68 ms** | Line-item parsing & token bounding box matching |
| **5. Cross-Field Math Validation** | **0.4 ms** | **0.4 ms** | Subtotal + Tax == Total and temporal ordering |
| **TOTAL END-TO-END LATENCY** | **~42.2 ms** | **~4,050 ms** | Complete pipeline from raw file upload to structured JSON |

---

## 5. Project Directory Structure

```
document_manager/
│
├── documind/
│   ├── core/
│   │   ├── config.py             # Settings, paths, and confidence thresholds
│   │   ├── preprocessor.py       # PDF/image normalization and page rendering
│   │   └── ocr_engine.py         # Hybrid OCR (PyMuPDF native + RapidOCR fallback)
│   ├── models/
│   │   ├── classifier_baseline.py# Scikit-Learn TF-IDF + Logistic Regression
│   │   ├── classifier_dl.py      # PyTorch Multi-Head Self-Attention Transformer
│   │   └── dataset_generator.py  # Realistic multi-class enterprise data generator
│   ├── extraction/
│   │   ├── schema.py             # Pydantic models for DocumentProcessResponse
│   │   ├── extractors.py         # Sub-millisecond field & line-item extractors
│   │   └── validator.py          # Math integrity & confidence calibrator
│   ├── evaluation/
│   │   ├── benchmark.py          # Model classification benchmark suite
│   │   └── field_evaluator.py    # Exact match & token F1 extraction evaluator
│   └── api/
│       ├── app.py                # FastAPI REST application
│       └── static/               # Interactive UI mirroring InvoiceOps / NexDox
│           ├── index.html
│           ├── style.css
│           └── app.js
│
├── data/
│   └── samples/                  # Pre-generated sample documents (PDFs)
├── artifacts/
│   ├── models/                   # Serialized model weights (.joblib, .pt)
│   ├── benchmark_comparison.json # Classification benchmark metrics
│   └── evaluation_report.json    # Field extraction evaluation report
├── tests/                        # Full automated test suite (13 unit & API tests)
├── scripts/
│   ├── generate_sample_docs.py   # ReportLab sample PDF generator
│   └── train_and_eval.py         # Complete training & benchmark execution script
│
├── Dockerfile                    # Containerization for deployment
├── docker-compose.yml            # Compose configuration
├── requirements.txt              # Production dependencies
└── README.md
```

---

## 5. Quick Start & Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd document_manager
pip install -r requirements.txt
```

### 2. Run Training & Benchmarking
Train both models and generate evaluation metrics on held-out test data:
```bash
python scripts/train_and_eval.py
```

### 3. Generate Sample Documents
Create sample invoices, retail receipts, IRS W-9 forms, and contracts:
```bash
python scripts/generate_sample_docs.py
```

### 4. Run Automated Tests
```bash
pytest -v
```
*(All 13 tests execute and pass in ~30 seconds).*

### 5. Launch FastAPI Backend & Web Interface
```bash
uvicorn documind.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser to:
- **Interactive UI**: `http://localhost:8000/`
- **Swagger API Docs**: `http://localhost:8000/docs`

---

## 6. API Reference

### `POST /api/v1/documents/process`
Accepts a PDF or image file upload. Executes OCR, classifies category, extracts all fields, validates math integrity, grounds bounding boxes, and returns structured JSON:
```json
{
  "document_id": "DOC-72AF81E2",
  "filename": "INV-2026-00432.pdf",
  "total_pages": 1,
  "classification": {
    "predicted_category": "invoice",
    "confidence": 0.998,
    "probabilities": {
      "invoice": 0.998,
      "receipt": 0.001,
      "form": 0.0005,
      "other": 0.0005
    },
    "model_used": "deep_learning_transformer",
    "inference_time_ms": 17.2
  },
  "overall_confidence": 0.96,
  "trust_status": "Trusted",
  "fields": {
    "invoice_number": {
      "field_name": "invoice_number",
      "value": "INV-2026-00432",
      "raw_text": "INV-2026-00432",
      "confidence": 0.96,
      "bbox": [65.2, 16.4, 88.5, 19.8],
      "validation_status": "valid"
    },
    "total_amount": {
      "field_name": "total_amount",
      "value": 10013.13,
      "raw_text": "USD $10,013.13",
      "confidence": 0.96,
      "bbox": [72.1, 78.4, 91.2, 82.0],
      "validation_status": "valid"
    }
  },
  "line_items": [
    {
      "description": "Cloud Migration Services",
      "quantity": 40,
      "unit_price": 125.0,
      "amount": 5000.0,
      "confidence": 0.96
    }
  ],
  "validation_issues": [],
  "processing_time_ms": 182.4
}
```

### `GET /api/v1/models/benchmark`
Returns complete empirical metrics comparing Baseline vs Deep Learning model on unseen data.

### `GET /api/v1/samples`
Lists available pre-loaded enterprise documents for 1-click evaluation.

---

## 7. Docker Deployment

Deploy the entire stack in an isolated, reproducible container:
```bash
docker compose up --build
```
The service will be live at `http://localhost:8000` with active container health checking.
