"""
DocuMind Core Configuration
Settings, supported document categories, confidence thresholds, and system constants.
"""

from pathlib import Path
from pydantic import BaseModel, Field

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DocuMindConfig(BaseModel):
    # Directory paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    SAMPLES_DIR: Path = SAMPLES_DIR
    ARTIFACTS_DIR: Path = ARTIFACTS_DIR
    MODELS_DIR: Path = MODELS_DIR

    # Document Classes
    CATEGORIES: list[str] = ["invoice", "receipt", "form", "other"]
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85   # >= 0.85 -> "Trusted"
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.70 # 0.70 - 0.84 -> "Needs review"
                                             # < 0.70 -> "Flagged"
                                             
    # OCR Settings
    DPI_RENDER: int = 150
    OCR_MAX_PAGES: int = 5
    
    # Model Artifact Paths
    BASELINE_MODEL_PATH: Path = MODELS_DIR / "baseline_tfidf_classifier.joblib"
    DL_MODEL_PATH: Path = MODELS_DIR / "dl_document_classifier.pt"
    DL_TOKENIZER_PATH: Path = MODELS_DIR / "dl_tokenizer"
    EVALUATION_REPORT_PATH: Path = ARTIFACTS_DIR / "evaluation_report.json"
    BENCHMARK_REPORT_PATH: Path = ARTIFACTS_DIR / "benchmark_comparison.json"
    
    # Device
    USE_CUDA_IF_AVAILABLE: bool = True


settings = DocuMindConfig()
