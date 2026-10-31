"""
Document Classification Models Unit Tests
"""

import pytest
from documind.models.classifier_baseline import BaselineDocumentClassifier
from documind.models.classifier_dl import DeepLearningDocumentClassifier


@pytest.fixture
def baseline_model():
    clf = BaselineDocumentClassifier()
    loaded = clf.load()
    assert loaded, "Baseline model failed to load from artifacts/models"
    return clf


@pytest.fixture
def dl_model():
    clf = DeepLearningDocumentClassifier()
    loaded = clf.load()
    assert loaded, "Deep Learning model failed to load from artifacts/models"
    return clf


def test_baseline_classifier_predictions(baseline_model):
    invoice_text = "INVOICE # INV-2026-00432 Total Due: $10,013.13 Subtotal: $9,250.00 Tax: $763.13 Due Date: Jun 19, 2026"
    res = baseline_model.predict(invoice_text)
    assert res.predicted_category == "invoice"
    assert res.confidence >= 0.50
    assert "invoice" in res.probabilities

    receipt_text = "STAR COFFEE RECEIPT # RCPT-84920 TOTAL AMOUNT $31.81 CASHIER POS-02 VISA 4092"
    res_rcpt = baseline_model.predict(receipt_text)
    assert res_rcpt.predicted_category == "receipt"


def test_dl_classifier_predictions(dl_model):
    form_text = "FORM W-9 Request for Taxpayer Identification Number and Certification Department of the Treasury IRS"
    res = dl_model.predict(form_text)
    assert res.predicted_category == "form"
    assert res.confidence >= 0.70
    assert res.inference_time_ms >= 0.0
    assert "form" in res.probabilities
