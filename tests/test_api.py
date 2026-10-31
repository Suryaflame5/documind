"""
FastAPI Endpoints Integration Tests
"""

import pytest
from fastapi.testclient import TestClient
from documind.api.app import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_benchmark_endpoint():
    resp = client.get("/api/v1/models/benchmark")
    assert resp.status_code == 200
    data = resp.json()
    assert "benchmark" in data
    assert "field_evaluation" in data
    assert "models" in data["benchmark"]


def test_samples_endpoint():
    resp = client.get("/api/v1/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert "samples" in data
    assert len(data["samples"]) >= 1


def test_process_sample_invoice():
    resp = client.get("/api/v1/samples/sample_invoice.pdf/process")
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"]["predicted_category"] == "invoice"
    assert "fields" in data
    assert data["fields"]["invoice_number"]["value"] == "INV-2026-00432"
    assert abs(data["fields"]["total_amount"]["value"] - 10013.13) < 0.05
    assert len(data["page_images_base64"]) >= 1


def test_classify_text_endpoint():
    payload = {
        "text": "INVOICE\nVendor: Summit Technologies LLC\nInvoice #: INV-9901\nBill To: Acme Co.\nTotal Due: $450.00\nPayment Terms: Net 30. Thank you for your business!",
        "model": "baseline"
    }
    resp = client.post("/api/v1/documents/classify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["predicted_category"] == "invoice"
    assert data["confidence"] > 0.50


def test_extract_fields_endpoint():
    payload = {
        "text": "RECEIPT # RCPT-9921 DATE: 2026-05-10 TOTAL AMOUNT: $42.50"
    }
    resp = client.post("/api/v1/documents/extract", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "fields" in data
    assert data["fields"]["document_id"]["value"] == "RCPT-9921"
    assert abs(data["fields"]["total_amount"]["value"] - 42.50) < 0.05
