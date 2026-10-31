"""
Hybrid OCR Engine
Combines high-fidelity native digital PDF text extraction with RapidOCR (ONNX Runtime)
for scanned images and raster documents. Captures token coordinates and confidence scores.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Any
import pymupdf
from documind.extraction.schema import OCRSpan

try:
    from rapidocr_onnxruntime import RapidOCR
    _RAPID_OCR_AVAILABLE = True
except ImportError:
    _RAPID_OCR_AVAILABLE = False


class OCREngine:
    def __init__(self):
        self.rapid_ocr = RapidOCR() if _RAPID_OCR_AVAILABLE else None

    def extract_from_page(self, page: pymupdf.Page, page_num: int = 1) -> Tuple[str, List[OCRSpan]]:
        """
        Extracts full text and word/line level bounding boxes from a single page.
        Employs native vector text extraction first; if page is scanned/empty, falls back to RapidOCR.
        """
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # 1. Check native digital text in PDF
        native_text = page.get_text("text").strip()
        words = page.get_text("words")  # returns list of (x0, y0, x1, y1, word, block_no, line_no, word_no)

        if len(native_text) >= 40 and len(words) >= 5:
            # Native text is rich and available
            spans: List[OCRSpan] = []
            for w in words:
                x0, y0, x1, y1, word_str = w[0], w[1], w[2], w[3], w[4]
                # Normalized coordinates (0.0 to 100.0% of page dimension) for easy front-end overlay
                norm_bbox = [
                    round((x0 / page_width) * 100.0, 2),
                    round((y0 / page_height) * 100.0, 2),
                    round((x1 / page_width) * 100.0, 2),
                    round((y1 / page_height) * 100.0, 2),
                ]
                spans.append(OCRSpan(
                    text=word_str,
                    bbox=norm_bbox,
                    confidence=0.99,
                    page=page_num
                ))
            return native_text, spans

        # 2. Scanned or image-only page: Fallback to RapidOCR
        if self.rapid_ocr is not None:
            zoom = 2.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_np = np.array(img)

            ocr_results, _ = self.rapid_ocr(img_np)
            if not ocr_results:
                return native_text or "", []

            spans: List[OCRSpan] = []
            extracted_lines = []
            for item in ocr_results:
                box_pts, text_line, score = item[0], item[1], float(item[2])
                extracted_lines.append(text_line)
                
                # box_pts has 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in rendered pix coords
                xs = [pt[0] / (page_width * zoom) * 100.0 for pt in box_pts]
                ys = [pt[1] / (page_height * zoom) * 100.0 for pt in box_pts]
                norm_bbox = [
                    round(min(xs), 2),
                    round(min(ys), 2),
                    round(max(xs), 2),
                    round(max(ys), 2)
                ]
                spans.append(OCRSpan(
                    text=text_line,
                    bbox=norm_bbox,
                    confidence=round(score, 3),
                    page=page_num
                ))

            full_ocr_text = "\n".join(extracted_lines)
            return full_ocr_text, spans

        return native_text, []

    def extract_from_document(self, doc: pymupdf.Document) -> Tuple[str, List[OCRSpan]]:
        """
        Processes all pages in a document and aggregates text and token spans.
        """
        all_text_blocks = []
        all_spans: List[OCRSpan] = []

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_text, page_spans = self.extract_from_page(page, page_num=page_num)
            all_text_blocks.append(page_text)
            all_spans.extend(page_spans)

        combined_text = "\n\n".join(all_text_blocks)
        return combined_text, all_spans
