"""
Document Preprocessor
Handles input validation, format normalization (PDF / Images), page rendering, and preview generation.
"""

import io
import base64
from pathlib import Path
from typing import Union, List, Tuple
import pymupdf  # PyMuPDF
from PIL import Image


class DocumentPreprocessor:
    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def load_document(self, file_source: Union[str, Path, bytes, io.BytesIO], filename: str = "document.pdf") -> pymupdf.Document:
        """
        Loads PDF or image input into a unified PyMuPDF Document object.
        """
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            suffix = path.suffix.lower()
            if suffix in [".pdf"]:
                return pymupdf.open(str(path))
            elif suffix in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
                img = Image.open(str(path))
                pdf_bytes = io.BytesIO()
                img.convert("RGB").save(pdf_bytes, format="PDF")
                return pymupdf.open(stream=pdf_bytes.getvalue(), filetype="pdf")
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        elif isinstance(file_source, bytes):
            # Check magic bytes for PDF vs Image
            if file_source.startswith(b"%PDF"):
                return pymupdf.open(stream=file_source, filetype="pdf")
            else:
                try:
                    img = Image.open(io.BytesIO(file_source))
                    pdf_bytes = io.BytesIO()
                    img.convert("RGB").save(pdf_bytes, format="PDF")
                    return pymupdf.open(stream=pdf_bytes.getvalue(), filetype="pdf")
                except Exception as e:
                    # Try opening as pymupdf directly
                    return pymupdf.open(stream=file_source, filetype=Path(filename).suffix.lstrip("."))

        elif isinstance(file_source, io.BytesIO):
            return self.load_document(file_source.getvalue(), filename=filename)
        else:
            raise TypeError(f"Unsupported source type: {type(file_source)}")

    def render_page_to_base64_png(self, page: pymupdf.Page, zoom: float = 1.5) -> str:
        """
        Renders a document page to a base64 encoded PNG for browser UI rendering.
        """
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def render_page_to_pil(self, page: pymupdf.Page, zoom: float = 2.0) -> Image.Image:
        """
        Renders a document page to a PIL Image for OCR processing.
        """
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
