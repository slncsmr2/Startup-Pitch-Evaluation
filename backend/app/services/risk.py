def detect_risk_flags(chunk_text: str, aggregate_score: float) -> list[str]:
    lowered = chunk_text.lower()
    flags: list[str] = []

    if aggregate_score < 5.5:
        flags.append("low-quality-signal")

    if "no competition" in lowered or "no competitor" in lowered:
        flags.append("competitive-landscape-missing")

    if "guaranteed" in lowered or "100%" in lowered:
        flags.append("overclaim-risk")

    if "soon" in lowered and "revenue" in lowered:
        flags.append("traction-evidence-weak")

    return flags
