"""
Validation & Confidence Calibration Unit Tests
"""

import pytest
from documind.extraction.validator import DocumentValidator
from documind.extraction.schema import (
    FieldExtraction,
    DocumentClassificationResult,
    LineItem
)


@pytest.fixture
def validator():
    return DocumentValidator()


def test_validator_mathematical_pass(validator):
    classification = DocumentClassificationResult(
        predicted_category="invoice",
        confidence=0.95,
        probabilities={"invoice": 0.95, "receipt": 0.05},
        model_used="baseline_tfidf"
    )

    fields = {
        "document_id": FieldExtraction(field_name="document_id", value="INV-001", raw_text="INV-001", confidence=0.95),
        "total_amount": FieldExtraction(field_name="total_amount", value=108.0, raw_text="$108.00", confidence=0.96),
        "subtotal": FieldExtraction(field_name="subtotal", value=100.0, raw_text="$100.00", confidence=0.95),
        "tax": FieldExtraction(field_name="tax", value=8.0, raw_text="$8.00", confidence=0.94),
    }

    line_items = [
        LineItem(description="Item 1", amount=100.0, confidence=0.95)
    ]

    conf, status, issues = validator.validate(classification, fields, line_items)
    assert status == "Trusted"
    assert conf >= 0.85
    assert len(issues) == 0


def test_validator_mathematical_discrepancy(validator):
    classification = DocumentClassificationResult(
        predicted_category="invoice",
        confidence=0.95,
        probabilities={"invoice": 0.95, "receipt": 0.05},
        model_used="baseline_tfidf"
    )

    # Total doesn't match subtotal + tax
    fields = {
        "document_id": FieldExtraction(field_name="document_id", value="INV-002", raw_text="INV-002", confidence=0.95),
        "total_amount": FieldExtraction(field_name="total_amount", value=500.0, raw_text="$500.00", confidence=0.96),
        "subtotal": FieldExtraction(field_name="subtotal", value=100.0, raw_text="$100.00", confidence=0.95),
        "tax": FieldExtraction(field_name="tax", value=8.0, raw_text="$8.00", confidence=0.94),
    }

    conf, status, issues = validator.validate(classification, fields, [])
    assert status in ["Needs review", "Flagged"]
    assert any("Math Discrepancy" in iss for iss in issues)
