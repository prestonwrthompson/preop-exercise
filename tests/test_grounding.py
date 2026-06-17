from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core import PatientSubmission
from grounding_check import check_issues_value_grounding
from triage_llm import ClinicalExtraction
from triage_rules import evaluate_triage

ROOT = Path(__file__).resolve().parent.parent

_ANTICOAG_PATTERN = re.compile(
    r"\b(apixaban|warfarin|rivaroxaban|eliquis|xarelto|coumadin|heparin|enoxaparin)\b",
    re.IGNORECASE,
)
_HP_PATTERN = re.compile(r"history|h&p|physical", re.IGNORECASE)
_CONSENT_PATTERN = re.compile(r"consent", re.IGNORECASE)
_PLAN_PATTERN = re.compile(r"perioperative|peri-op|anticoag", re.IGNORECASE)


def _heuristic_extraction(submission: PatientSubmission) -> ClinicalExtraction:
    """Approximate LLM extraction from submission text for offline replay."""

    hp_docs: list[dict[str, object]] = []
    consent_docs: list[dict[str, object]] = []
    plan_docs: list[dict[str, object]] = []

    for index, document in enumerate(submission.documents or []):
        doc_type = document.type or ""
        text = document.text or ""
        combined = f"{doc_type} {text}"
        if _HP_PATTERN.search(combined):
            hp_docs.append({"document_index": index, "date": document.date})
        if _CONSENT_PATTERN.search(combined):
            signed = "signed" in text.lower() and "unsigned" not in text.lower()
            consent_docs.append(
                {
                    "document_index": index,
                    "date": document.date,
                    "is_signed": signed,
                }
            )
        if _PLAN_PATTERN.search(combined):
            clear = bool(
                re.search(
                    r"\b(hold|resume|stop|restart|bridge|days before|days after)\b",
                    text,
                    re.IGNORECASE,
                )
            )
            plan_docs.append({"document_index": index, "is_clear": clear})

    medications: list[dict[str, object]] = []
    for index, medication in enumerate(submission.medications or []):
        name = medication.name or ""
        is_anticoag = bool(_ANTICOAG_PATTERN.search(name))
        medications.append({"medication_index": index, "is_anticoagulant": is_anticoag})

    return ClinicalExtraction.model_validate(
        {
            "history_and_physical_documents": hp_docs,
            "surgical_consent_documents": consent_docs,
            "anticoagulation_plan_documents": plan_docs,
            "medications": medications,
        }
    )


def test_bp_exclusion_evidence_is_grounded() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {"procedure_risk": "MODERATE", "procedure_date": "2026-03-04"},
            "vitals": [
                {
                    "type": "blood_pressure",
                    "systolic": 184,
                    "diastolic": 111,
                    "date": "2026-02-27T10:12:00Z",
                    "source": "Primary care visit",
                },
                {"type": "temperature", "value_f": 98.8, "date": "2026-02-27T10:15:00Z"},
            ],
            "labs": [{"code": "CBC", "effective_at": "2026-02-24T08:10:00Z"}],
            "documents": [
                {"type": "H&P Note", "date": "2026-02-20", "text": "H&P completed."},
                {"type": "Consent Form", "date": "2026-02-25", "text": "Signed consent."},
            ],
        }
    )
    extraction = _heuristic_extraction(submission)
    output = evaluate_triage(submission, extraction)
    payload = submission.model_dump()

    assert check_issues_value_grounding(payload, output)
    bp_issue = next(i for i in output.issues if i.category == "ACUTE_SAFETY_EXCLUSION")
    assert "Primary care visit" in bp_issue.evidence.details
    assert "2026-02-27" in bp_issue.evidence.details


def test_anticoag_evidence_is_grounded() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {"procedure_risk": "LOW", "procedure_date": "2026-03-01"},
            "medications": [{"name": "apixaban", "active": True}],
            "documents": [
                {
                    "type": "Perioperative Medication Plan",
                    "date": "2026-02-22",
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

    assert check_issues_value_grounding(submission.model_dump(), output)
    issue = next(i for i in output.issues if i.category == "ANTICOAGULATION_MANAGEMENT")
    assert issue.evidence.source == "medications[0]"
    assert "apixaban" in issue.evidence.details


def test_missing_consent_evidence_is_grounded() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {"procedure_risk": "LOW", "procedure_date": "2026-03-12"},
            "vitals": [
                {"type": "blood_pressure", "systolic": 120, "diastolic": 73, "date": "2026-03-07"},
                {"type": "temperature", "value_f": 99.2, "date": "2026-03-07"},
            ],
            "labs": [{"code": "CBC", "effective_at": "2026-03-04T10:10:00Z"}],
            "documents": [
                {"type": "History/Physical (H&P)", "date": "2026-03-01", "text": "H&P note."},
            ],
        }
    )
    extraction = ClinicalExtraction(
        history_and_physical_documents=[{"document_index": 0, "date": "2026-03-01"}],
        surgical_consent_documents=[],
    )
    output = evaluate_triage(submission, extraction)

    assert check_issues_value_grounding(submission.model_dump(), output)
    issue = next(i for i in output.issues if "Consent" in i.description)
    assert "History/Physical" in issue.evidence.details


def test_missing_lab_evidence_is_grounded() -> None:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {"procedure_risk": "HIGH", "procedure_date": "2026-03-10"},
            "vitals": [
                {"type": "blood_pressure", "systolic": 118, "diastolic": 74, "date": "2026-03-05"},
                {"type": "temperature", "value_f": 98.4, "date": "2026-03-05"},
            ],
            "labs": [
                {
                    "code": "HBA1C",
                    "display": "Hemoglobin A1c",
                    "effective_at": "2026-02-09T07:40:00Z",
                }
            ],
            "documents": [
                {"type": "H&P", "date": "2026-03-01", "text": "H&P completed."},
                {"type": "Consent", "date": "2026-03-04", "text": "Signed consent."},
            ],
        }
    )
    extraction = _heuristic_extraction(submission)
    output = evaluate_triage(submission, extraction)

    assert check_issues_value_grounding(submission.model_dump(), output)
    cmp_issue = next(i for i in output.issues if "CMP" in i.description)
    assert "HBA1C" in cmp_issue.evidence.details


@pytest.mark.parametrize("case_id", [f"case_{i:05d}" for i in range(50)])
def test_eval_cases_grounded_with_heuristic_extraction(case_id: str) -> None:
    report_path = ROOT / "data" / "eval_report.json"
    if not report_path.exists():
        pytest.skip("eval_report.json not available")

    report = json.loads(report_path.read_text())
    record = next((rec for rec in report["records"] if rec["case_id"] == case_id), None)
    if record is None:
        pytest.skip(f"{case_id} not in eval report")

    submission = PatientSubmission.model_validate(record["submission"])
    extraction = _heuristic_extraction(submission)
    output = evaluate_triage(submission, extraction)

    assert check_issues_value_grounding(record["submission"], output), (
        f"{case_id} issues not grounded: "
        + str([i.evidence.details for i in output.issues])
    )
