"""
DocuMind Extraction Schemas & Pydantic Data Contracts
Standardized data structures for document OCR, classification, and field extraction.
"""

from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 1

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "page": self.page}


class OCRSpan(BaseModel):
    text: str
    bbox: List[float] # [x0, y0, x1, y1]
    confidence: float = 1.0
    page: int = 1


class FieldExtraction(BaseModel):
    field_name: str
    value: Any
    raw_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[List[float]] = None
    page: int = 1
    validation_status: str = "valid"  # "valid", "warning", "error", "unverified"
    validation_message: Optional[str] = None


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: float
    confidence: float = 0.90
    bbox: Optional[List[float]] = None


class DocumentClassificationResult(BaseModel):
    predicted_category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    model_used: str  # "baseline_tfidf" or "deep_learning_transformer"
    inference_time_ms: float = 0.0


class DocumentProcessResponse(BaseModel):
    document_id: str
    filename: str
    total_pages: int
    classification: DocumentClassificationResult
    overall_confidence: float
    trust_status: str  # "Trusted", "Needs review", "Flagged"
    fields: Dict[str, FieldExtraction]
    line_items: List[LineItem] = []
    validation_issues: List[str] = []
    processing_time_ms: float
    extracted_text: str
    ocr_spans: List[OCRSpan] = []
    page_images_base64: List[str] = []  # base64 encoded PNG pages for interactive sidecar view
