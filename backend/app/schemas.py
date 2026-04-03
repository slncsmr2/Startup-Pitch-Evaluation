from pydantic import BaseModel, Field


class PitchVideoInput(BaseModel):
    file_name: str = Field(..., description="Pitch video file name")
    file_format: str = Field(default="mp4", description="Video format: mp4 or mov")
    duration_sec: int = Field(default=60, ge=5)
    transcript_text: str = Field(default="", description="Optional ASR transcript")


class SlideInput(BaseModel):
    title: str = Field(default="")
    content: str = Field(default="")


class UserDetails(BaseModel):
    founder_name: str = Field(default="")
    startup_name: str = Field(default="")
    sector: str = Field(default="")
    stage: str = Field(default="")


class PitchInput(BaseModel):
    title: str = Field(..., description="Startup name or pitch title")
    transcript: str = Field(default="", description="Full pitch transcript")
    language_hint: str = Field(default="en-ta", description="Language mix hint")
    presenter_profile: dict = Field(default_factory=dict)
    slide_text: list[str] = Field(default_factory=list)
    video: PitchVideoInput | None = Field(default=None)
    slides: list[SlideInput] = Field(default_factory=list)
    user_details: UserDetails | None = Field(default=None)


class MetricScore(BaseModel):
    name: str
    score: float
    rationale: str


class ChunkReport(BaseModel):
    chunk_id: int
    start_sec: int
    end_sec: int
    text_metrics: list[MetricScore]
    av_metrics: list[MetricScore]
    attention: dict[str, float]
    risk_flags: list[str]
    aggregate_score: float


class EvaluationSummary(BaseModel):
    overall_score: float
    confidence_score: float
    investment_band: str
    language_detected: str
    scoring_mode: str = "heuristic"
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    processing_option: str = "unknown"
    processing_notes: list[str] = Field(default_factory=list)


class DashboardSeriesPoint(BaseModel):
    label: str
    value: float


class InvestorDashboard(BaseModel):
    quantitative_scores: list[DashboardSeriesPoint]
    modality_weights: list[DashboardSeriesPoint]
    risk_distribution: list[DashboardSeriesPoint]


class EvaluationResponse(BaseModel):
    request_id: str
    summary: EvaluationSummary
    chunk_reports: list[ChunkReport]
    dashboard: InvestorDashboard


class BatchEvaluationRequest(BaseModel):
    pitches: list[PitchInput] = Field(default_factory=list)


class BatchEvaluationResponse(BaseModel):
    evaluations: list[EvaluationResponse]
