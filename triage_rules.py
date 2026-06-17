"""Deterministic triage rules driven by submission data and LLM extractions."""

from __future__ import annotations

from datetime import date, datetime

from core import (
    Decision,
    PatientSubmission,
    TriageIssue,
    TriageIssueEvidence,
    TriageOutput,
)
from triage_llm import ClinicalExtraction


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        parsed_date = _parse_date(value)
        if parsed_date is None:
            return None
        return datetime.combine(parsed_date, datetime.min.time())


def _days_before_procedure(procedure_date: date, event_date: date) -> int:
    return (procedure_date - event_date).days


def _issue(
    category: str,
    description: str,
    source: str,
    details: str,
) -> TriageIssue:
    return TriageIssue(
        category=category,
        description=description,
        evidence=TriageIssueEvidence(source=source, details=details),
    )


def _document_date(
    submission: PatientSubmission,
    document_index: int,
    extracted_date: str | None,
) -> date | None:
    parsed = _parse_date(extracted_date)
    if parsed is not None:
        return parsed
    documents = submission.documents or []
    if document_index < 0 or document_index >= len(documents):
        return None
    return _parse_date(documents[document_index].date)


def _lab_code_matches(code: str | None, test: str) -> bool:
    normalized = (code or "").upper()
    return test.upper() in normalized


def _latest_vital_index(submission: PatientSubmission, vital_type: str) -> int | None:
    best_index: int | None = None
    best_date: date | None = None
    for index, vital in enumerate(submission.vitals or []):
        if (vital.type or "").lower() != vital_type:
            continue
        parsed = _parse_date(vital.date)
        if parsed is None:
            continue
        if best_date is None or parsed > best_date:
            best_date = parsed
            best_index = index
    return best_index


def _latest_lab_index(submission: PatientSubmission, test: str) -> int | None:
    best_index: int | None = None
    best_at: datetime | None = None
    for index, lab in enumerate(submission.labs or []):
        if not _lab_code_matches(lab.code, test):
            continue
        parsed = _parse_datetime(lab.effective_at)
        if parsed is None:
            continue
        if best_at is None or parsed > best_at:
            best_at = parsed
            best_index = index
    return best_index


def _evaluate_missing_procedure_fields(submission: PatientSubmission) -> list[TriageIssue]:
    issues: list[TriageIssue] = []
    procedure = submission.procedure

    if procedure is None or procedure.procedure_date is None:
        issues.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                "Missing procedure date",
                "procedure.procedure_date",
                "procedure.procedure_date is null",
            )
        )

    if procedure is None or procedure.procedure_risk is None:
        issues.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                "Missing procedure risk",
                "procedure.procedure_risk",
                "procedure.procedure_risk is null",
            )
        )

    return issues


def _evaluate_acute_safety(submission: PatientSubmission) -> list[TriageIssue]:
    issues: list[TriageIssue] = []

    bp_index = _latest_vital_index(submission, "blood_pressure")
    if bp_index is None:
        issues.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                "Missing latest blood pressure",
                "vitals",
                "No blood_pressure vital with valid date found",
            )
        )
    else:
        vital = submission.vitals[bp_index]
        systolic = getattr(vital, "systolic", None)
        diastolic = getattr(vital, "diastolic", None)
        if systolic is not None and diastolic is not None:
            if systolic >= 180 or diastolic >= 110:
                issues.append(
                    _issue(
                        "ACUTE_SAFETY_EXCLUSION",
                        "Blood pressure meets exclusion threshold",
                        f"vitals[{bp_index}]",
                        (
                            f"latest BP systolic={systolic}, diastolic={diastolic}; "
                            "threshold systolic>=180 or diastolic>=110"
                        ),
                    )
                )

    temp_index = _latest_vital_index(submission, "temperature")
    if temp_index is None:
        issues.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                "Missing latest temperature",
                "vitals",
                "No temperature vital with valid date found",
            )
        )
    else:
        vital = submission.vitals[temp_index]
        value_f = getattr(vital, "value_f", None)
        if value_f is not None and value_f > 100.4:
            issues.append(
                _issue(
                    "ACUTE_SAFETY_EXCLUSION",
                    "Temperature exceeds exclusion threshold",
                    f"vitals[{temp_index}]",
                    f"latest temperature value_f={value_f}; threshold is > 100.4",
                )
            )

    return issues


def _evaluate_required_testing(submission: PatientSubmission) -> list[TriageIssue]:
    issues: list[TriageIssue] = []
    procedure = submission.procedure
    assert procedure is not None
    assert procedure.procedure_date is not None
    assert procedure.procedure_risk is not None

    procedure_date = _parse_date(procedure.procedure_date)
    if procedure_date is None:
        return issues

    risk = procedure.procedure_risk
    cbc_window = 30 if risk in {"LOW", "MODERATE"} else 14

    cbc_index = _latest_lab_index(submission, "CBC")
    if cbc_index is None:
        issues.append(
            _issue(
                "REQUIRED_TESTING",
                "Missing required CBC",
                "labs",
                f"No CBC result with valid effective_at found for procedure_risk {risk}",
            )
        )
    else:
        lab = submission.labs[cbc_index]
        effective_at = _parse_datetime(lab.effective_at)
        if effective_at is None:
            issues.append(
                _issue(
                    "REQUIRED_TESTING",
                    "Missing required CBC",
                    "labs",
                    f"No CBC result with valid effective_at found for procedure_risk {risk}",
                )
            )
        else:
            days = _days_before_procedure(procedure_date, effective_at.date())
            if days > cbc_window:
                issues.append(
                    _issue(
                        "REQUIRED_TESTING",
                        "CBC outside required window",
                        f"labs[{cbc_index}]",
                        (
                            f"CBC effective_at {lab.effective_at} vs procedure_date "
                            f"{procedure_date.isoformat()} ({days} days prior; must be within "
                            f"{cbc_window})"
                        ),
                    )
                )

    if risk == "HIGH":
        cmp_index = _latest_lab_index(submission, "CMP")
        if cmp_index is None:
            issues.append(
                _issue(
                    "REQUIRED_TESTING",
                    "Missing required CMP",
                    "labs",
                    f"No CMP result with valid effective_at found for procedure_risk {risk}",
                )
            )
        else:
            lab = submission.labs[cmp_index]
            effective_at = _parse_datetime(lab.effective_at)
            if effective_at is None:
                issues.append(
                    _issue(
                        "REQUIRED_TESTING",
                        "Missing required CMP",
                        "labs",
                        f"No CMP result with valid effective_at found for procedure_risk {risk}",
                    )
                )
            else:
                days = _days_before_procedure(procedure_date, effective_at.date())
                if days > 14:
                    issues.append(
                        _issue(
                            "REQUIRED_TESTING",
                            "CMP outside required window",
                            f"labs[{cmp_index}]",
                            (
                                f"CMP effective_at {lab.effective_at} vs procedure_date "
                                f"{procedure_date.isoformat()} ({days} days prior; must be within 14)"
                            ),
                        ),
                    )

    return issues


def _evaluate_documentation_from_extraction(
    submission: PatientSubmission,
    extraction: ClinicalExtraction,
) -> list[TriageIssue]:
    issues: list[TriageIssue] = []
    procedure = submission.procedure
    if procedure is None or procedure.procedure_date is None:
        return issues

    procedure_date = _parse_date(procedure.procedure_date)
    if procedure_date is None:
        return issues

    hp_entries: list[tuple[int, date, int]] = []
    valid_hp_exists = False
    for hp in extraction.history_and_physical_documents:
        document_date = _document_date(submission, hp.document_index, hp.date)
        if document_date is None:
            continue
        days = _days_before_procedure(procedure_date, document_date)
        if days <= 30:
            valid_hp_exists = True
        elif days > 30:
            hp_entries.append((hp.document_index, document_date, days))

    if not valid_hp_exists:
        if hp_entries:
            index, document_date, days = max(hp_entries, key=lambda item: item[1])
            issues.append(
                _issue(
                    "REQUIRED_DOCUMENTATION",
                    "H&P outside 30-day window",
                    f"documents[{index}]",
                    (
                        f"H&P date {document_date.isoformat()} vs procedure_date "
                        f"{procedure_date.isoformat()} ({days} days prior; must be within 30)"
                    ),
                )
            )
        else:
            issues.append(
                _issue(
                    "REQUIRED_DOCUMENTATION",
                    "History and Physical document missing",
                    "documents",
                    "No History and Physical document with valid date found",
                )
            )

    if not extraction.surgical_consent_documents:
        issues.append(
            _issue(
                "REQUIRED_DOCUMENTATION",
                "Surgical Consent document missing",
                "documents",
                "No Surgical Consent document found",
            )
        )
        return issues

    for consent in extraction.surgical_consent_documents:
        if consent.is_signed is False:
            documents = submission.documents or []
            snippet = ""
            if 0 <= consent.document_index < len(documents):
                snippet = (documents[consent.document_index].text or "").strip()
            if len(snippet) > 80:
                snippet = f"{snippet[:77]}..."
            issues.append(
                _issue(
                    "REQUIRED_DOCUMENTATION",
                    "Consent document text does not clearly indicate signed consent",
                    f"documents[{consent.document_index}]",
                    (
                        "Consent document text does not clearly indicate signed consent: "
                        f"{snippet}"
                    ),
                )
            )
            return issues

    return issues


def _anticoag_evidence_source(
    submission: PatientSubmission,
    med_index: int,
    plan_doc_index: int | None,
) -> str:
    if plan_doc_index is None:
        return "documents"

    documents = submission.documents or []
    if plan_doc_index < 0 or plan_doc_index >= len(documents):
        return "documents"

    medication = submission.medications[med_index]
    med_name = (medication.name or "").strip().lower()
    doc_text = (documents[plan_doc_index].text or "").lower()
    if med_name and med_name in doc_text:
        return f"documents[{plan_doc_index}]"
    return "documents"


def _evaluate_anticoagulation_from_extraction(
    submission: PatientSubmission,
    extraction: ClinicalExtraction,
) -> list[TriageIssue]:
    issues: list[TriageIssue] = []
    medications = submission.medications or []
    anticoag_flags = {
        item.medication_index: item.is_anticoagulant
        for item in extraction.medications
    }

    active_anticoag_index: int | None = None
    for index, medication in enumerate(medications):
        if not anticoag_flags.get(index, False):
            continue
        if medication.active is None:
            issues.append(
                _issue(
                    "MISSING_REQUIRED_DATA",
                    "Unknown anticoagulant active status",
                    f"medications[{index}]",
                    (
                        f"Medication {medication.name} has active=null; "
                        "cannot determine if currently taking"
                    ),
                )
            )
            return issues
        if medication.active:
            active_anticoag_index = index

    if active_anticoag_index is None:
        return issues

    plan_is_clear = any(plan.is_clear is True for plan in extraction.anticoagulation_plan_documents)
    if plan_is_clear:
        return issues

    plan_doc_index = None
    if extraction.anticoagulation_plan_documents:
        plan_doc_index = extraction.anticoagulation_plan_documents[-1].document_index

    source = _anticoag_evidence_source(submission, active_anticoag_index, plan_doc_index)
    issues.append(
        _issue(
            "ANTICOAGULATION_MANAGEMENT",
            "Missing perioperative anticoagulation plan",
            source,
            (
                f"Active anticoagulant medication present (medications[{active_anticoag_index}]) "
                "but no clear perioperative plan document found"
            ),
        )
    )
    return issues


def evaluate_triage(
    submission: PatientSubmission,
    extraction: ClinicalExtraction,
) -> TriageOutput:
    """Build the final triage output from submission data and extracted clinical facts."""

    issues: list[TriageIssue] = []
    issues.extend(_evaluate_missing_procedure_fields(submission))
    issues.extend(_evaluate_anticoagulation_from_extraction(submission, extraction))

    procedure = submission.procedure
    if procedure is not None and procedure.procedure_date is not None:
        issues.extend(_evaluate_acute_safety(submission))
        issues.extend(_evaluate_documentation_from_extraction(submission, extraction))

    if (
        procedure is not None
        and procedure.procedure_date is not None
        and procedure.procedure_risk is not None
    ):
        issues.extend(_evaluate_required_testing(submission))

    return TriageOutput(
        decision=_derive_decision(issues),
        issues=issues,
        explanation=_build_explanation(issues),
    )


def _derive_decision(issues: list[TriageIssue]) -> Decision:
    categories = {issue.category for issue in issues}
    if "ACUTE_SAFETY_EXCLUSION" in categories:
        return "NOT_CLEARED"
    if issues:
        return "NEEDS_FOLLOW_UP"
    return "READY"


def _build_explanation(issues: list[TriageIssue]) -> str:
    if not issues:
        return (
            "All required documentation, testing, anticoagulation planning, "
            "and safety checks are satisfied."
        )
    parts = [f"{issue.category}: {issue.description}" for issue in issues]
    return " | ".join(parts)
