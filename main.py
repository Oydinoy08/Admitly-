"""
Admitly — AI Personal Statement Evaluator
Backend API service built with FastAPI.

Run locally:
    pip install fastapi uvicorn pydantic
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Admitly API",
    description=(
        "AI-powered backend for Admitly — a platform that teaches students "
        "to write personal statements from scratch and evaluates finished "
        "essays against a 10-point admissions rubric."
    ),
    version="1.0.0",
    contact={"name": "Oydinoy Abdumaxmudjonova"},
)

# Allow the frontend (served from any origin during development) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MIN_ESSAY_WORDS = 40
MAX_ESSAY_WORDS = 1200


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class TargetUniversity(str, Enum):
    """Optional hint used to slightly tailor the academic-motivation check."""

    UNSPECIFIED = "unspecified"
    OTHER = "other"


class EssaySubmission(BaseModel):
    """Payload sent by the client when requesting an essay evaluation."""

    essay_text: str = Field(
        ...,
        min_length=1,
        description="Full text of the student's personal statement.",
        examples=[
            "Growing up in a small town, I never imagined that a single "
            "broken laptop would set the course of my academic life..."
        ],
    )
    word_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Word count reported by the client. If omitted, the server "
            "computes it from essay_text."
        ),
    )
    target_university: str | None = Field(
        default=None,
        max_length=200,
        description="Name of the university or program the student is applying to.",
        examples=["MIT", "Nazarbayev University"],
    )

    @field_validator("essay_text")
    @classmethod
    def essay_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("essay_text must not be blank")
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "essay_text": (
                    "Growing up in a small town, I never imagined that a "
                    "single broken laptop would set the course of my "
                    "academic life. Fixing it taught me that curiosity, "
                    "not equipment, is what drives real learning..."
                ),
                "word_count": 48,
                "target_university": "MIT",
            }
        }


class CriterionScore(BaseModel):
    """A single 0-10 score for one evaluation criterion."""

    clarity: float = Field(..., ge=0, le=10, description="Clarity & Structure score (0-10).")
    structure: float = Field(..., ge=0, le=10, description="Paragraph/organizational structure score (0-10).")
    impact: float = Field(..., ge=0, le=10, description="Personal Story & Impact score (0-10).")
    grammar: float = Field(..., ge=0, le=10, description="Grammar & Tone score (0-10).")


class EvaluationResult(BaseModel):
    """Response returned after evaluating a submitted essay."""

    overall_score: float = Field(..., ge=0, le=10, description="Average of all criterion scores, 0-10.")
    clarity: float = Field(..., ge=0, le=10, description="Clarity & Structure score (0-10).")
    structure: float = Field(..., ge=0, le=10, description="Structure score (0-10).")
    impact: float = Field(..., ge=0, le=10, description="Personal Story & Impact score (0-10).")
    grammar: float = Field(..., ge=0, le=10, description="Grammar & Tone score (0-10).")
    word_count: int = Field(..., ge=0, description="Word count of the evaluated essay.")
    feedback_list: List[str] = Field(
        ..., description="Actionable feedback: strengths first, then areas to improve."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "overall_score": 7.8,
                "clarity": 8.1,
                "structure": 7.4,
                "impact": 8.6,
                "grammar": 7.2,
                "word_count": 214,
                "feedback_list": [
                    "Strength: The personal story is vivid and draws the reader in immediately.",
                    "Strength: The essay's structure moves logically from anecdote to reflection.",
                    "Improve: Tie the closing paragraph more explicitly back to your academic goals.",
                ],
            }
        }


class EvaluationCriterion(BaseModel):
    """Describes one of the 10 rubric criteria used by Admitly."""

    id: int = Field(..., description="1-indexed position of the criterion in the rubric.")
    name: str = Field(..., description="Short name of the criterion.")
    description: str = Field(..., description="What this criterion measures.")


class CriteriaResponse(BaseModel):
    """Full list of the 10 rubric criteria."""

    criteria: List[EvaluationCriterion]


class HealthResponse(BaseModel):
    """Simple liveness/readiness payload."""

    status: str = Field(..., description="'ok' when the service is healthy.")
    service: str = Field(..., description="Name of the service.")
    version: str = Field(..., description="Deployed API version.")


# --------------------------------------------------------------------------- #
# Evaluation logic
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

_MOTIVATION_KEYWORDS = {
    "study", "research", "university", "degree", "career", "future",
    "passion", "goal", "major", "academic", "program", "learn", "curiosity",
}
_NARRATIVE_MARKERS = {
    "when", "remember", "realized", "learned", "taught", "moment", "experience",
}
_FIRST_PERSON = {"i", "my", "me", "myself"}


def _clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    """Restrict a value to the closed interval [low, high]."""
    return max(low, min(high, value))


def _tokenize_words(text: str) -> List[str]:
    """Extract lowercase word tokens from essay text."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _split_sentences(text: str) -> List[str]:
    """Split essay text into non-empty sentences."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _split_paragraphs(text: str) -> List[str]:
    """Split essay text into non-empty paragraphs (blank-line separated)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def evaluate_essay_logic(essay_text: str) -> dict:
    """
    Analyze essay text and produce 0-10 scores for four rubric dimensions.

    The scoring is a transparent, deterministic heuristic (word counts,
    keyword presence, sentence/paragraph structure) rather than a call to an
    external LLM, so the endpoint stays fast, free to run, and fully
    explainable. Swap the body of this function for a real model call
    (e.g. an LLM prompt) without changing its signature or return shape.

    Args:
        essay_text: Raw personal statement text submitted by the student.

    Returns:
        A dict with keys: clarity, structure, impact, grammar, word_count,
        overall_score, feedback_list.
    """
    words = _tokenize_words(essay_text)
    word_count = len(words)
    sentences = _split_sentences(essay_text)
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sentence_count
    paragraphs = _split_paragraphs(essay_text)
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count else 0.0

    # --- Clarity: rewards a balanced average sentence length (~17 words) ---
    sentence_ideal = 1 - min(abs(avg_sentence_len - 17) / 17, 1)
    clarity = _clamp(5 + sentence_ideal * 5)

    # --- Structure: rewards multiple, reasonably sized paragraphs ---
    paragraph_bonus = _clamp((len(paragraphs) - 1) * 1.2, 0, 4)
    structure = _clamp(5 + paragraph_bonus)

    # --- Impact: first-person density + narrative language ---
    first_person_count = sum(1 for w in words if w in _FIRST_PERSON)
    narrative_count = sum(1 for w in words if w in _NARRATIVE_MARKERS)
    fp_density = (first_person_count / word_count * 60) if word_count else 0
    impact = _clamp(4 + min(fp_density, 3) + min(narrative_count * 0.5, 3))

    # --- Grammar & tone: lexical diversity + sane length as a proxy ---
    length_bonus = _clamp((word_count - 60) / 200, 0, 1.5)
    grammar = _clamp(4 + lexical_diversity * 6 + length_bonus)

    # --- Academic motivation keyword count feeds into feedback text only ---
    motivation_hits = sum(1 for w in words if w in _MOTIVATION_KEYWORDS)

    overall_score = round((clarity + structure + impact + grammar) / 4, 1)

    scores = {
        "clarity": round(clarity, 1),
        "structure": round(structure, 1),
        "impact": round(impact, 1),
        "grammar": round(grammar, 1),
    }

    feedback_list = _build_feedback(scores, motivation_hits, paragraphs)

    return {
        **scores,
        "word_count": word_count,
        "overall_score": overall_score,
        "feedback_list": feedback_list,
    }


def _build_feedback(scores: dict, motivation_hits: int, paragraphs: list) -> List[str]:
    """Turn raw scores into 3 human-readable feedback bullets."""
    strengths_bank = {
        "clarity": "Strength: The essay reads clearly, with well-paced sentences.",
        "structure": "Strength: The essay is organized into distinct, purposeful paragraphs.",
        "impact": "Strength: The personal story feels genuine and draws the reader in.",
        "grammar": "Strength: The language is varied and the tone fits the genre.",
    }
    improve_bank = {
        "clarity": "Improve: Break down longer sentences to make the argument easier to follow.",
        "structure": "Improve: Split the essay into more focused paragraphs, one idea each.",
        "impact": "Improve: Anchor the essay in one specific moment instead of a general summary.",
        "grammar": "Improve: Vary sentence length further to keep the tone consistently engaging.",
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_two = ranked[:2]
    lowest = ranked[-1]

    feedback = [strengths_bank[key] for key, _ in top_two]
    feedback.append(improve_bank[lowest[0]])

    if motivation_hits == 0:
        feedback.append(
            "Improve: Make your academic motivation more explicit — mention why this "
            "field or program matters to you."
        )
    if len(paragraphs) <= 1:
        feedback.append(
            "Improve: Break the essay into multiple paragraphs to separate story, "
            "reflection, and conclusion."
        )

    return feedback


# --------------------------------------------------------------------------- #
# Rubric data
# --------------------------------------------------------------------------- #

EVALUATION_CRITERIA: List[EvaluationCriterion] = [
    EvaluationCriterion(id=1, name="Clarity & Structure", description="The essay has a clear introduction, development, and conclusion."),
    EvaluationCriterion(id=2, name="Personal Story & Impact", description="The personal experience is conveyed authentically and memorably."),
    EvaluationCriterion(id=3, name="Academic Motivation", description="The student's interest in the chosen field is clearly justified."),
    EvaluationCriterion(id=4, name="Grammar & Tone", description="The writing is free of errors and matches the tone of the genre."),
    EvaluationCriterion(id=5, name="Originality & Voice", description="The author's unique perspective and voice come through."),
    EvaluationCriterion(id=6, name="Specificity & Detail", description="Concrete examples and details are used instead of generalities."),
    EvaluationCriterion(id=7, name="Coherence & Flow", description="Transitions between ideas are natural and consistent."),
    EvaluationCriterion(id=8, name="Relevance to Program", description="The essay demonstrates fit with the target program or university."),
    EvaluationCriterion(id=9, name="Critical Thinking", description="The author draws thoughtful conclusions from their experience."),
    EvaluationCriterion(id=10, name="Overall Persuasiveness", description="The essay is convincing and memorable as a whole."),
]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """
    Report whether the API service is running.

    Returns:
        HealthResponse: A static payload confirming the service is healthy.
    """
    return HealthResponse(status="ok", service="admitly-api", version=app.version)


@app.get(
    "/api/v1/criteria",
    response_model=CriteriaResponse,
    tags=["Evaluation"],
    summary="List the 10-point evaluation criteria",
)
async def get_criteria() -> CriteriaResponse:
    """
    Return the full list of rubric criteria used to evaluate essays.

    Returns:
        CriteriaResponse: All 10 named criteria with their descriptions.
    """
    return CriteriaResponse(criteria=EVALUATION_CRITERIA)


@app.post(
    "/api/v1/evaluate-essay",
    response_model=EvaluationResult,
    status_code=status.HTTP_200_OK,
    tags=["Evaluation"],
    summary="Evaluate a personal statement essay",
)
async def evaluate_essay(submission: EssaySubmission) -> EvaluationResult:
    """
    Analyze a submitted essay and return scores plus written feedback.

    The essay is checked for length, keyword presence, and sentence/paragraph
    structure using `evaluate_essay_logic`, then mapped onto four rubric
    dimensions (clarity, structure, impact, grammar) and an overall score.

    Args:
        submission: The essay text and optional metadata to evaluate.

    Raises:
        HTTPException: 422 if the essay is shorter than the minimum word
            count required for a meaningful evaluation.

    Returns:
        EvaluationResult: Scores (0-10) for each dimension, an overall
        score, and a list of feedback strings.
    """
    computed_word_count = len(_tokenize_words(submission.essay_text))

    if computed_word_count < MIN_ESSAY_WORDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"essay_text must contain at least {MIN_ESSAY_WORDS} words "
                f"for a meaningful evaluation (got {computed_word_count})."
            ),
        )

    if computed_word_count > MAX_ESSAY_WORDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"essay_text exceeds the maximum of {MAX_ESSAY_WORDS} words "
                f"(got {computed_word_count})."
            ),
        )

    result = evaluate_essay_logic(submission.essay_text)

    return EvaluationResult(
        overall_score=result["overall_score"],
        clarity=result["clarity"],
        structure=result["structure"],
        impact=result["impact"],
        grammar=result["grammar"],
        word_count=result["word_count"],
        feedback_list=result["feedback_list"],
    )


# --------------------------------------------------------------------------- #
# Local entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
