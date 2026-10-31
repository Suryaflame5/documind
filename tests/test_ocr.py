"""
OCR & Preprocessing Unit Tests
"""

import pytest
from pathlib import Path
from documind.core.config import settings
from documind.core.preprocessor import DocumentPreprocessor
from documind.core.ocr_engine import OCREngine


@pytest.fixture
def sample_pdf():
    pdf_path = settings.SAMPLES_DIR / "sample_invoice.pdf"
    assert pdf_path.exists(), f"Sample invoice missing at {pdf_path}"
    return pdf_path


def test_document_preprocessor_load(sample_pdf):
    preprocessor = DocumentPreprocessor()
    doc = preprocessor.load_document(sample_pdf)
    assert len(doc) >= 1
    
    # Test rendering to base64
    b64_img = preprocessor.render_page_to_base64_png(doc[0])
    assert isinstance(b64_img, str)
    assert len(b64_img) > 100


def test_ocr_engine_extraction(sample_pdf):
    preprocessor = DocumentPreprocessor()
    ocr = OCREngine()
    
    doc = preprocessor.load_document(sample_pdf)
    text, spans = ocr.extract_from_document(doc)
    
    assert len(text) > 50
    assert "SUMMIT" in text or "INVOICE" in text or "INV-2026-00432" in text
    assert len(spans) > 5
    
    # Check normalized coordinate ranges [0, 100]
    first_span = spans[0]
    bbox = first_span.bbox
    assert 0.0 <= bbox[0] <= 100.0
    assert 0.0 <= bbox[1] <= 100.0
    assert 0.0 <= bbox[2] <= 100.0
    assert 0.0 <= bbox[3] <= 100.0
