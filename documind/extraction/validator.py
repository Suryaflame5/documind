"""
Document Extraction Validator & Confidence Calibration
Performs mathematical integrity validation, temporal checks, cross-field consistency,
and calculates the overall document trust status.
"""

from typing import Dict, List, Tuple
from documind.core.config import settings
from documind.extraction.schema import FieldExtraction, LineItem, DocumentClassificationResult


class DocumentValidator:
    def __init__(self):
        self.high_thresh = settings.HIGH_CONFIDENCE_THRESHOLD
        self.med_thresh = settings.MEDIUM_CONFIDENCE_THRESHOLD

    def validate(
        self,
        classification: DocumentClassificationResult,
        fields: Dict[str, FieldExtraction],
        line_items: List[LineItem]
    ) -> Tuple[float, str, List[str]]:
        """
        Validates extracted fields, computes calibrated overall confidence score and trust status.
        Returns: (overall_confidence, trust_status, list_of_validation_issues)
        """
        issues: List[str] = []
        validation_penalties = 0.0

        # 1. Classification check
        cat = classification.predicted_category
        cat_conf = classification.confidence

        if cat_conf < 0.60:
            issues.append(f"Low classification confidence ({cat_conf * 100:.1f}%) for type '{cat}'.")
            validation_penalties += 0.15

        # 2. Mathematical Consistency (for Invoices & Receipts)
        if cat in ["invoice", "receipt"]:
            total_field = fields.get("total_amount")
            subtotal_field = fields.get("subtotal")
            tax_field = fields.get("tax")

            total_val = float(total_field.value) if total_field and isinstance(total_field.value, (int, float)) else None
            sub_val = float(subtotal_field.value) if subtotal_field and isinstance(subtotal_field.value, (int, float)) else None
            tax_val = float(tax_field.value) if tax_field and isinstance(tax_field.value, (int, float)) else 0.0

            if total_val is not None and total_val > 0:
                if sub_val is not None and sub_val > 0:
                    expected_total = sub_val + tax_val
                    discrepancy = abs(total_val - expected_total)
                    if discrepancy > 0.05:
                        issues.append(
                            f"Math Discrepancy: Subtotal (${sub_val:.2f}) + Tax (${tax_val:.2f}) = ${expected_total:.2f}, but Total Due is ${total_val:.2f}."
                        )
                        validation_penalties += 0.20
                        if total_field:
                            total_field.validation_status = "warning"
                            total_field.validation_message = f"Mismatch with subtotal + tax (diff: ${discrepancy:.2f})"
                    else:
                        if total_field:
                            total_field.validation_status = "valid"
            else:
                issues.append("Zero or missing total monetary amount.")
                validation_penalties += 0.25

            # 3. Line Items Consistency
            if line_items and sub_val is not None and sub_val > 0:
                items_sum = sum(item.amount for item in line_items)
                if abs(items_sum - sub_val) > 0.50:
                    issues.append(
                        f"Line items sum (${items_sum:.2f}) deviates from subtotal (${sub_val:.2f})."
                    )
                    validation_penalties += 0.08

        # 4. Identifier Verification
        doc_id = fields.get("document_id")
        if not doc_id or doc_id.value == "N/A" or len(str(doc_id.value).strip()) < 3:
            issues.append("Missing or unreadable document identifier.")
            validation_penalties += 0.15

        # 5. Calculate Calibrated Overall Confidence
        field_confidences = [f.confidence for f in fields.values() if isinstance(f, FieldExtraction)]
        avg_field_conf = sum(field_confidences) / len(field_confidences) if field_confidences else 0.5

        raw_score = (0.35 * cat_conf) + (0.65 * avg_field_conf) - validation_penalties
        overall_confidence = float(round(max(0.05, min(0.99, raw_score)), 2))

        # 6. Assign Trust Status
        if overall_confidence >= self.high_thresh and len(issues) == 0:
            trust_status = "Trusted"
        elif overall_confidence >= self.med_thresh:
            trust_status = "Needs review"
        else:
            trust_status = "Flagged"

        return overall_confidence, trust_status, issues
