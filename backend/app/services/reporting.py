def build_feedback(overall_score: float, text_avg: float, av_avg: float, top_risks: list[str]) -> dict:
    strengths: list[str] = []
    weaknesses: list[str] = []
    suggestions: list[str] = []

    if text_avg >= 7:
        strengths.append("Strong articulation of problem and market context")
    else:
        weaknesses.append("Narrative around problem-market fit needs sharper framing")
        suggestions.append("Open with customer pain backed by one concrete metric")

    if av_avg >= 7:
        strengths.append("Confident delivery and visual support")
    else:
        weaknesses.append("Presentation confidence and pacing can improve")
        suggestions.append("Tighten slide text and slow down key transitions")

    if overall_score >= 8:
        strengths.append("Investor readiness appears high")
    elif overall_score < 6:
        suggestions.append("Add traction proof points and a clearer business model story")

    if "competitive-landscape-missing" in top_risks:
        weaknesses.append("Competitive positioning is not clearly defended")
        suggestions.append("Add competitor matrix with clear differentiation")

    if "overclaim-risk" in top_risks:
        weaknesses.append("Claims may appear inflated to investors")
        suggestions.append("Replace absolute claims with verifiable evidence")

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
    }


def build_investor_dashboard(
    quantitative_scores: list[dict],
    modality_attention: dict[str, float],
    risk_counts: dict[str, int],
) -> dict:
    return {
        "quantitative_scores": [
            {"label": item["name"], "value": round(item["score"], 2)}
            for item in quantitative_scores
        ],
        "modality_weights": [
            {"label": key, "value": round(value, 2)} for key, value in modality_attention.items()
        ],
        "risk_distribution": [
            {"label": key, "value": float(value)} for key, value in sorted(risk_counts.items())
        ],
    }
