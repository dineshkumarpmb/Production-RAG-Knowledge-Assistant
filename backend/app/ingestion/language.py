from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # langdetect is random by default; make results reproducible


def detect_language(text: str, *, min_chars: int = 20, sample_chars: int = 5000) -> str | None:
    """Return an ISO 639-1 code (e.g. "en"), or None if there is too little text to tell."""
    sample = text[:sample_chars].strip()
    if len(sample) < min_chars:
        return None
    try:
        return str(detect(sample))
    except LangDetectException:
        return None
