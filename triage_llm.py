"""LLM extraction of clinical facts for deterministic triage."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from core import PatientSubmission, load_prompt

EXTRACTION_SYSTEM_PROMPT = load_prompt("clinical_extraction.txt")


class ExtractedHistoryAndPhysical(BaseModel):
    document_index: int
    date: str | None = None


class ExtractedSurgicalConsent(BaseModel):
    document_index: int
    date: str | None = None
    is_signed: bool | None = None


class ExtractedAnticoagulationPlan(BaseModel):
    document_index: int
    is_clear: bool | None = None


class ExtractedMedication(BaseModel):
    medication_index: int
    is_anticoagulant: bool


class ClinicalExtraction(BaseModel):
    """Structured facts extracted by the LLM; not a triage decision."""

    history_and_physical_documents: list[ExtractedHistoryAndPhysical] = Field(
        default_factory=list
    )
    surgical_consent_documents: list[ExtractedSurgicalConsent] = Field(
        default_factory=list
    )
    anticoagulation_plan_documents: list[ExtractedAnticoagulationPlan] = Field(
        default_factory=list
    )
    medications: list[ExtractedMedication] = Field(default_factory=list)


def clinical_extraction_json_schema() -> dict[str, object]:
    return ClinicalExtraction.model_json_schema()


def build_extraction_user_prompt(submission: dict[str, object]) -> str:
    sections = [
        "Extract clinical documentation and medication facts from this submission.",
        "Return JSON only, using the schema described in the instructions.",
        "Submission JSON:",
        json.dumps(submission, sort_keys=True),
    ]
    return "\n".join(sections)


def extract_clinical_facts(
    submission: PatientSubmission,
    *,
    model: str,
) -> ClinicalExtraction:
    """Call the LLM to extract clinical facts from the submission."""

    from openai import OpenAI

    submission_payload = submission.model_dump()
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=EXTRACTION_SYSTEM_PROMPT,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_extraction_user_prompt(submission_payload),
                    }
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "clinical_extraction",
                "schema": clinical_extraction_json_schema(),
                "strict": False,
            }
        },
    )

    return ClinicalExtraction.model_validate_json(response.output_text)
