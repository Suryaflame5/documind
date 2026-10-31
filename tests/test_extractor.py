"""
Field Extraction Unit Tests
"""

import pytest
from documind.extraction.extractors import DocumentFieldExtractor
from documind.extraction.schema import OCRSpan


@pytest.fixture
def extractor():
    return DocumentFieldExtractor()


def test_reference_invoice_extraction(extractor):
    raw_text = """
    Summit Technologies LLC
    345 Innovation Way, Suite 200, Denver, CO 80202
    
    INVOICE
    Invoice #: INV-2026-00432
    Invoice Date: May 20, 2026
    Due Date: Jun 19, 2026

    Bill To:
    Acme Co.
    123 Market Street, Floor 4, San Francisco, CA 94105

    DESCRIPTION                          QTY    UNIT PRICE        AMOUNT
    Cloud Migration Services              40    $  125.00    $   5000.00
    Security Assessment                    1    $ 2500.00    $   2500.00
    Data Integration & ETL                 1    $ 1750.00    $   1750.00

    Subtotal:        $   9250.00
    Tax (8.25%):     $    763.13
    TOTAL DUE:       USD $ 10013.13
    """

    spans = [
        OCRSpan(text="INV-2026-00432", bbox=[70.0, 15.0, 85.0, 18.0], confidence=0.99, page=1),
        OCRSpan(text="10013.13", bbox=[75.0, 80.0, 88.0, 85.0], confidence=0.98, page=1),
    ]

    fields, line_items = extractor.extract_all(raw_text, spans)

    # Document ID
    assert fields["document_id"].value == "INV-2026-00432"
    assert fields["document_id"].bbox is not None

    # Dates
    assert "May 20, 2026" in fields["invoice_date"].value
    assert "Jun 19, 2026" in fields["due_date"].value

    # Amounts
    assert abs(float(fields["total_amount"].value) - 10013.13) < 0.01
    assert abs(float(fields["subtotal"].value) - 9250.00) < 0.01
    assert abs(float(fields["tax"].value) - 763.13) < 0.01

    # Currency
    assert fields["currency"].value == "USD"

    # Line Items
    assert len(line_items) >= 3
    assert line_items[0].description == "Cloud Migration Services"
    assert line_items[0].amount == 5000.0
