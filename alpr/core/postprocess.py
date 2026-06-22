"""Postprocesamiento y validación de placas."""

import re


MERCOSUR_PATTERNS = [
    re.compile(r"^[A-Z]{3}\d[A-Z]\d{2}$"),   # Brasil/Argentina Mercosur: ABC1D23
    re.compile(r"^[A-Z]{2}\d{3}[A-Z]{2}$"),  # Uruguay Mercosur: AB123CD
    re.compile(r"^[A-Z]{4}\d{3}$"),          # Paraguay Mercosur: ABCD123
    re.compile(r"^[A-Z]{3}\d{4}$"),          # Brasil antiguo: ABC1234
]


def clean_plate_text(raw_text: str) -> str:
    """Normaliza texto: mayúsculas, solo alfanumérico."""
    text = raw_text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def is_mercosur_format(text: str) -> bool:
    """Valida formato Mercosur (acepta variaciones regionales)."""
    return any(pattern.match(text) for pattern in MERCOSUR_PATTERNS)


def normalize_confidence(score: float) -> float:
    """Asegura que la confianza esté en [0, 1]."""
    return float(max(0.0, min(1.0, score)))


def post_process(raw_text: str, score: float) -> dict:
    """Limpia y valida una lectura OCR."""
    text = clean_plate_text(raw_text)
    valid = is_mercosur_format(text)
    return {
        "text": text,
        "raw_text": raw_text,
        "confidence": normalize_confidence(score),
        "valid_format": valid,
    }
