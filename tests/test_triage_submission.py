from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core import (
    PatientSubmission,
    TriageIssue,
    TriageIssueEvidence,
    TriageOutput,
    triage_submission,
)
from triage_llm import EXTRACTION_SYSTEM_PROMPT, ClinicalExtraction
from triage_rules import evaluate_triage


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "history_and_physical_documents": [],
                "surgical_consent_documents": [],
                "anticoagulation_plan_documents": [],
                "medications": [],
            }
        ),
    )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "date": "2026-01-25",
            },
            {
                "type": "temperature",
                "value_f": 98.6,
                "date": "2026-01-25",
            },
        ],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {
                "type": "history_and_physical",
                "date": "2026-01-20",
                "text": "History and physical completed.",
            },
            {
                "type": "surgical_consent",
                "date": "2026-01-22",
                "text": "Signed surgical consent.",
            },
        ],
    }


def test_triage_submission_calls_extraction_llm(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "history_and_physical_documents": [
                    {"document_index": 0, "date": "2026-01-20"}
                ],
                "surgical_consent_documents": [
                    {"document_index": 1, "date": "2026-01-22", "is_signed": True}
                ],
                "anticoagulation_plan_documents": [],
                "medications": [],
            }
        )
    )

    output = triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == EXTRACTION_SYSTEM_PROMPT


def test_evaluate_triage_flags_missing_procedure_date_and_anticoag() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {
                "procedure_risk": "LOW",
                "procedure_date": None,
            },
            "medications": [{"name": "apixaban", "active": True}],
            "documents": [
                {
                    "type": "Perioperative Medication Plan",
                    "text": "Follow up with cardiology for peri-op recommendations.",
                }
            ],
        }
    )
    extraction = ClinicalExtraction(
        medications=[{"medication_index": 0, "is_anticoagulant": True}],
        anticoagulation_plan_documents=[{"document_index": 0, "is_clear": False}],
    )

    output = evaluate_triage(submission, extraction)

    categories = {issue.category for issue in output.issues}
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert categories == {"MISSING_REQUIRED_DATA", "ANTICOAGULATION_MANAGEMENT"}


def test_evaluate_triage_not_cleared_for_fever() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {
                "procedure_risk": "LOW",
                "procedure_date": "2026-03-03",
            },
            "vitals": [
                {
                    "type": "temperature",
                    "value_f": 101.0,
                    "date": "2026-02-26",
                },
                {
                    "type": "blood_pressure",
                    "systolic": 120,
                    "diastolic": 80,
                    "date": "2026-02-26",
                },
            ],
            "labs": [{"code": "CBC", "effective_at": "2026-02-23"}],
            "documents": [],
        }
    )
    extraction = ClinicalExtraction(
        history_and_physical_documents=[{"document_index": 0, "date": "2026-02-21"}],
        surgical_consent_documents=[{"document_index": 1, "date": "2026-02-25", "is_signed": True}],
    )

    output = evaluate_triage(submission, extraction)

    assert output.decision == "NOT_CLEARED"
    assert any(issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in output.issues)


def test_evaluate_triage_stale_hp_when_no_valid_hp_in_extraction() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {
                "procedure_risk": "LOW",
                "procedure_date": "2026-03-08",
            },
            "vitals": [
                {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-01"},
                {"type": "temperature", "value_f": 98.6, "date": "2026-03-01"},
            ],
            "labs": [{"code": "CBC", "effective_at": "2026-03-01T09:10:00Z"}],
            "documents": [
                {"type": "History/Physical (H&P)", "date": "2026-01-22", "text": "H&P completed."},
                {"type": "Consent - Elective Procedure", "date": "2026-03-02", "text": "Signed consent."},
            ],
        }
    )
    extraction = ClinicalExtraction(
        history_and_physical_documents=[{"document_index": 0, "date": "2026-01-22"}],
        surgical_consent_documents=[{"document_index": 1, "date": "2026-03-02", "is_signed": True}],
        medications=[],
    )

    output = evaluate_triage(submission, extraction)

    assert any(
        issue.category == "REQUIRED_DOCUMENTATION" and issue.evidence.source == "documents[0]"
        for issue in output.issues
    )


def test_evaluate_triage_unknown_anticoag_active_status_only() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {
                "procedure_risk": "LOW",
                "procedure_date": "2026-03-02",
            },
            "vitals": [
                {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-02-25"},
                {"type": "temperature", "value_f": 99.0, "date": "2026-02-25"},
            ],
            "labs": [{"code": "CBC", "effective_at": "2026-02-22T09:10:00Z"}],
            "medications": [
                {"name": "lisinopril", "active": True},
                {"name": "warfarin", "active": None},
            ],
            "documents": [
                {"type": "H&P Note", "date": "2026-02-20", "text": "H&P completed."},
                {"type": "Procedure Consent Form", "date": "2026-02-24", "text": "Signed consent."},
            ],
        }
    )
    extraction = ClinicalExtraction(
        history_and_physical_documents=[{"document_index": 0, "date": "2026-02-20"}],
        surgical_consent_documents=[{"document_index": 1, "date": "2026-02-24", "is_signed": True}],
        medications=[
            {"medication_index": 0, "is_anticoagulant": False},
            {"medication_index": 1, "is_anticoagulant": True},
        ],
    )

    output = evaluate_triage(submission, extraction)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [issue.category for issue in output.issues] == ["MISSING_REQUIRED_DATA"]
