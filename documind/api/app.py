"""
DocuMind FastAPI Application
Exposes RESTful endpoints for document OCR, classification, structured extraction,
mathematical validation, model benchmarking, and serves the interactive web UI.
"""

import time
import uuid
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from documind.core.config import settings
from documind.core.preprocessor import DocumentPreprocessor
from documind.core.ocr_engine import OCREngine
from documind.models.classifier_baseline import BaselineDocumentClassifier
from documind.models.classifier_dl import DeepLearningDocumentClassifier
from documind.extraction.extractors import DocumentFieldExtractor
from documind.extraction.validator import DocumentValidator
from documind.extraction.schema import (
    DocumentProcessResponse,
    DocumentClassificationResult,
    FieldExtraction,
    LineItem,
    OCRSpan
)

app = FastAPI(
    title="DocuMind Enterprise Document Intelligence System",
    description="High-precision document understanding, classification, and traceable field extraction API.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines and models
preprocessor = DocumentPreprocessor(dpi=settings.DPI_RENDER)
ocr_engine = OCREngine()
field_extractor = DocumentFieldExtractor()
validator = DocumentValidator()

baseline_clf = BaselineDocumentClassifier()
baseline_clf.load()

dl_clf = DeepLearningDocumentClassifier()
dl_clf.load()


class ClassifyRequest(BaseModel):
    text: str
    model: str = "deep_learning"  # "baseline" or "deep_learning"


class ExtractRequest(BaseModel):
    text: str


@app.get("/api/v1/health")
def health_check():
    import torch
    return {
        "status": "healthy",
        "service": "DocuMind Document Intelligence",
        "version": "1.0.0",
        "device": "cuda" if torch.cuda.is_available() and settings.USE_CUDA_IF_AVAILABLE else "cpu",
        "models_loaded": {
            "baseline_tfidf": baseline_clf.is_trained,
            "deep_learning_transformer": dl_clf.is_trained
        }
    }


@app.get("/api/v1/models/benchmark")
def get_model_benchmark():
    """
    Returns full evaluation benchmark metrics for Baseline vs Deep Learning models.
    """
    if not settings.BENCHMARK_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found. Run scripts/train_and_eval.py first.")
    
    with open(settings.BENCHMARK_REPORT_PATH, "r", encoding="utf-8") as f:
        import json
        bench_data = json.load(f)
        
    eval_data = {}
    if settings.EVALUATION_REPORT_PATH.exists():
        with open(settings.EVALUATION_REPORT_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            
    return {
        "benchmark": bench_data,
        "field_evaluation": eval_data
    }


@app.get("/api/v1/samples")
def list_samples():
    """
    Returns available enterprise sample documents for 1-click test runs.
    """
    samples = []
    if settings.SAMPLES_DIR.exists():
        for p in settings.SAMPLES_DIR.glob("*.pdf"):
            samples.append({
                "filename": p.name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
                "description": {
                    "sample_invoice.pdf": "Reference Commercial Invoice (Summit Technologies to Acme Co., $10,013.13)",
                    "sample_receipt.pdf": "Point-of-Sale Retail Receipt (Star Coffee & Bakery, $31.81)",
                    "sample_form.pdf": "IRS Form W-9 Request for Taxpayer ID",
                    "sample_contract.pdf": "Master Services Agreement (MSA) Legal Covenant"
                }.get(p.name, "Enterprise Business Document")
            })
    return {"samples": samples}


def _process_document_bytes(file_bytes: bytes, filename: str, model_type: str = "deep_learning") -> DocumentProcessResponse:
    t_start = time.time()
    doc_id = f"DOC-{uuid.uuid4().hex[:8].upper()}"

    # 1. Load document and render preview images
    doc = preprocessor.load_document(file_bytes, filename=filename)
    page_images_base64 = []
    for page in doc:
        page_images_base64.append(preprocessor.render_page_to_base64_png(page))

    # 2. Extract OCR Text and Bounding Box Spans
    extracted_text, ocr_spans = ocr_engine.extract_from_document(doc)

    # 3. Document Classification
    if model_type == "baseline" and baseline_clf.is_trained:
        classification = baseline_clf.predict(extracted_text)
    elif dl_clf.is_trained:
        classification = dl_clf.predict(extracted_text)
    elif baseline_clf.is_trained:
        classification = baseline_clf.predict(extracted_text)
    else:
        classification = DocumentClassificationResult(
            predicted_category="invoice",
            confidence=0.85,
            probabilities={"invoice": 0.85, "receipt": 0.05, "form": 0.05, "other": 0.05},
            model_used="heuristic_fallback"
        )

    # 4. Field Extraction with Bounding Box Grounding
    fields, line_items = field_extractor.extract_all(extracted_text, ocr_spans)

    # 5. Validation and Confidence Calibration
    overall_conf, trust_status, issues = validator.validate(classification, fields, line_items)

    total_time_ms = round((time.time() - t_start) * 1000.0, 2)

    return DocumentProcessResponse(
        document_id=doc_id,
        filename=filename,
        total_pages=len(doc),
        classification=classification,
        overall_confidence=overall_conf,
        trust_status=trust_status,
        fields=fields,
        line_items=line_items,
        validation_issues=issues,
        processing_time_ms=total_time_ms,
        extracted_text=extracted_text,
        ocr_spans=ocr_spans,
        page_images_base64=page_images_base64
    )


@app.post("/api/v1/documents/process", response_model=DocumentProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    model: str = Query("deep_learning", description="Classification model: baseline or deep_learning")
):
    """
    Processes an uploaded PDF or image: runs OCR, classification, field extraction, validation,
    and returns full structured JSON with visual bounding box coordinates.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return _process_document_bytes(file_bytes, filename=file.filename or "uploaded_doc.pdf", model_type=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {str(e)}")


@app.get("/api/v1/samples/{sample_name}/process", response_model=DocumentProcessResponse)
def process_sample_document(
    sample_name: str,
    model: str = Query("deep_learning", description="Classification model: baseline or deep_learning")
):
    """
    Processes a pre-loaded sample document from data/samples.
    """
    sample_path = settings.SAMPLES_DIR / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample document not found: {sample_name}")

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    return _process_document_bytes(file_bytes, filename=sample_name, model_type=model)


@app.post("/api/v1/documents/classify", response_model=DocumentClassificationResult)
def classify_text(req: ClassifyRequest):
    """
    Direct endpoint to classify raw document text.
    """
    if req.model == "baseline" and baseline_clf.is_trained:
        return baseline_clf.predict(req.text)
    elif dl_clf.is_trained:
        return dl_clf.predict(req.text)
    elif baseline_clf.is_trained:
        return baseline_clf.predict(req.text)
    else:
        raise HTTPException(status_code=500, detail="No classification models are currently trained.")


@app.post("/api/v1/documents/extract")
def extract_fields_endpoint(req: ExtractRequest):
    """
    Extracts structured fields directly from raw text.
    """
    fields, line_items = field_extractor.extract_all(req.text, spans=[])
    return {
        "fields": fields,
        "line_items": line_items
    }


# Static Web UI Mounting
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(static_dir / "index.html"))
