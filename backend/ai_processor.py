"""Stage 1 AI document classification and structured extraction for MedTranslate.

This module implements the first stage of the MedTranslate AI pipeline:
1. Inspecting the validated output from ``backend.document_processor``.
2. Routing PDF text, scanned PDFs, or images to the Gemini API via ``google-genai``.
3. Classifying the document (``prescription``, ``lab_report``, or ``unknown``).
4. Extracting only explicitly present prescription or lab report fields into
   Pydantic-validated JSON without diagnosing or interpreting medical results.

A downstream medical retrieval (RAG) and patient-explanation layer can consume
the structured output produced here in a subsequent stage.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger("medtranslate.ai_processor")

DEFAULT_GEMINI_MODEL: str = "gemini-3.1-flash-lite"
DEFAULT_LANGUAGE: str = "English"
DEFAULT_DISCLAIMER: str = (
    "MedTranslate provides educational explanations of your medical documents "
    "and is not a substitute for professional medical advice or diagnosis."
)

DocumentType = Literal["prescription", "lab_report", "unknown"]
InputMode = Literal["pdf_text", "scanned_pdf", "image"]

_NULL_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "null",
        "none",
        "n/a",
        "na",
        "not available",
        "not applicable",
        "not specified",
        "not mentioned",
        "not provided",
        "unavailable",
        "unknown",
        "-",
        "--",
    }
)

_SYSTEM_INSTRUCTION: str = """You are a careful medical document extraction assistant for MedTranslate.

Your ONLY task in this stage is document classification and verbatim structured extraction.

Strict safety and accuracy rules:
1. Classify the document strictly as one of: "prescription", "lab_report", or "unknown".
   - Use "prescription" for medication orders/prescriptions listing prescribed drugs.
   - Use "lab_report" for laboratory diagnostic test results.
   - Use "unknown" if the document is unreadable, non-medical, or neither a prescription nor a lab report.
2. Extract ONLY information that is explicitly written in the document.
3. NEVER invent, guess, or infer missing fields. Use null for any unavailable field.
4. Do NOT diagnose any condition.
5. Do NOT recommend starting, changing, or stopping any medication.
6. Do NOT interpret whether lab results are normal, abnormal, high, low, or clinically significant.
7. For "summary", provide only a brief, factual, non-interpretive description of what document type was identified and what items are listed, written in the requested target language. If the document is unreadable or unknown, state that factually.
8. Return valid JSON matching the requested schema.
"""


class AIProcessorError(RuntimeError):
    """Base exception for AI processing stage failures."""


class AIConfigurationError(AIProcessorError):
    """Raised when the Gemini SDK or API key is missing or misconfigured."""


class DocumentInputError(AIProcessorError):
    """Raised when ``processed_doc`` is malformed or unreadable."""


class GeminiAPIError(AIProcessorError):
    """Raised when the Gemini API call fails or returns unusable output."""


def _normalize_optional_text(value: Any) -> str | None:
    """Normalize optional string fields, turning blanks or placeholders into None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if cleaned.lower() in _NULL_PLACEHOLDERS:
        return None
    return cleaned


class ExtractedMedicine(BaseModel):
    """Structured prescription medication item extracted verbatim from a document."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(default=None, description="Medicine name as written on the prescription.")
    strength: str | None = Field(default=None, description="Medication strength (e.g., 500 mg), or null.")
    dosage: str | None = Field(default=None, description="Dose amount per intake (e.g., 1 tablet), or null.")
    frequency: str | None = Field(default=None, description="How often taken (e.g., twice daily), or null.")
    timing: str | None = Field(default=None, description="Timing instructions (e.g., after meals), or null.")
    duration: str | None = Field(default=None, description="Course duration (e.g., 5 days), or null.")

    @field_validator("name", "strength", "dosage", "frequency", "timing", "duration", mode="before")
    @classmethod
    def _clean_fields(cls, value: Any) -> str | None:
        return _normalize_optional_text(value)

    def has_any_value(self) -> bool:
        """Return True if at least one field contains extracted data."""
        return any(
            getattr(self, field_name) is not None
            for field_name in ("name", "strength", "dosage", "frequency", "timing", "duration")
        )


class ExtractedLabTest(BaseModel):
    """Structured laboratory test item extracted verbatim from a lab report."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(default=None, description="Laboratory test name as written.")
    result: str | None = Field(default=None, description="Observed test result value, or null.")
    unit: str | None = Field(default=None, description="Measurement unit (e.g., g/dL), or null.")
    reference_range: str | None = Field(
        default=None,
        description="Laboratory reference range exactly as printed, or null.",
    )
    date: str | None = Field(default=None, description="Test or report date if available, or null.")

    @field_validator("name", "result", "unit", "reference_range", "date", mode="before")
    @classmethod
    def _clean_fields(cls, value: Any) -> str | None:
        return _normalize_optional_text(value)

    def has_any_value(self) -> bool:
        """Return True if at least one field contains extracted data."""
        return any(
            getattr(self, field_name) is not None
            for field_name in ("name", "result", "unit", "reference_range", "date")
        )


class DocumentExtractionResult(BaseModel):
    """Validated Stage 1 output returned by Gemini classification and extraction."""

    model_config = ConfigDict(extra="ignore")

    document_type: Literal["prescription", "lab_report", "unknown"] = "unknown"
    summary: str = ""
    medicines: list[ExtractedMedicine] = Field(default_factory=list)
    tests: list[ExtractedLabTest] = Field(default_factory=list)

    @field_validator("document_type", mode="before")
    @classmethod
    def _normalize_document_type(cls, value: Any) -> DocumentType:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"prescription", "lab_report", "unknown"}:
                return normalized  # type: ignore[return-value]
        return "unknown"

    @field_validator("summary", mode="before")
    @classmethod
    def _normalize_summary(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @model_validator(mode="after")
    def _filter_empty_items(self) -> DocumentExtractionResult:
        self.medicines = [med for med in self.medicines if med.has_any_value()]
        self.tests = [test for test in self.tests if test.has_any_value()]
        if self.document_type == "unknown":
            self.medicines = []
            self.tests = []
        return self


class MedTranslateStage1Response(BaseModel):
    """Predictable Stage 1 pipeline contract compatible with ``app.py`` and future RAG."""

    document_type: Literal["prescription", "lab_report", "unknown"] = "unknown"
    summary: str = ""
    medicines: list[ExtractedMedicine] = Field(default_factory=list)
    tests: list[ExtractedLabTest] = Field(default_factory=list)
    questions_for_doctor: list[str] = Field(default_factory=list)
    disclaimer: str = DEFAULT_DISCLAIMER
    stage: Literal["extraction"] = "extraction"
    input_mode: Literal["pdf_text", "scanned_pdf", "image"] | None = None
    language: str = DEFAULT_LANGUAGE
    error: str | None = None


def detect_input_mode(processed_doc: dict[str, Any]) -> InputMode:
    """Determine whether ``processed_doc`` is PDF text, a scanned PDF, or an image."""
    if not isinstance(processed_doc, dict):
        raise DocumentInputError("processed_doc must be a dictionary.")

    file_type = (processed_doc.get("file_type") or "").strip().lower()
    is_scanned = bool(processed_doc.get("is_scanned"))
    extracted_text = processed_doc.get("extracted_text")

    if file_type == "image":
        return "image"

    if file_type == "pdf":
        if not is_scanned and isinstance(extracted_text, str) and extracted_text.strip():
            return "pdf_text"
        return "scanned_pdf"

    raise DocumentInputError(f"Unsupported or missing file_type in processed_doc: {file_type!r}.")


def _read_binary_file(filepath: Any, label: str) -> bytes:
    """Read binary bytes from disk for scanned PDFs or image fallbacks."""
    if not isinstance(filepath, str) or not filepath.strip():
        raise DocumentInputError(f"Missing valid filepath for {label}.")
    if not os.path.isfile(filepath):
        raise DocumentInputError(f"Document file for {label} was not found on disk: {filepath}")

    try:
        with open(filepath, "rb") as file_handle:
            data = file_handle.read()
    except OSError as exc:
        raise DocumentInputError(f"Failed to read {label} from disk: {exc}") from exc

    if not data:
        raise DocumentInputError(f"Document file for {label} is empty.")
    return data


def _build_extraction_prompt(language: str, input_mode: InputMode) -> str:
    """Construct the Stage 1 extraction prompt for Gemini."""
    source_hint = {
        "pdf_text": "The input below is text extracted from a PDF document.",
        "scanned_pdf": "The attached file is a scanned or image-based PDF document.",
        "image": "The attached file is an uploaded medical document image.",
    }[input_mode]

    return (
        f"{source_hint}\n"
        f"Target language for the brief factual summary: {language}.\n\n"
        "Tasks:\n"
        "1. Classify the document as 'prescription', 'lab_report', or 'unknown'.\n"
        "2. Extract only the details explicitly present in the document:\n"
        "   - If it is a prescription, populate `medicines` with items having: "
        "name, strength, dosage, frequency, timing, duration.\n"
        "   - If it is a lab_report, populate `tests` with items having: "
        "name, result, unit, reference_range, date.\n"
        "   - If it is unknown, leave both `medicines` and `tests` as empty lists [].\n"
        "3. Use null for any field not explicitly visible or stated.\n"
        "4. Write a short, strictly factual `summary` describing what was observed "
        "in the document without any diagnosis, clinical interpretation, or medication advice.\n"
    )


def _build_gemini_contents(
    processed_doc: dict[str, Any],
    input_mode: InputMode,
    language: str,
    types_module: Any,
) -> list[Any]:
    """Assemble the multimodal or text payload expected by ``google-genai``."""
    prompt = _build_extraction_prompt(language=language, input_mode=input_mode)

    if input_mode == "pdf_text":
        extracted_text = str(processed_doc.get("extracted_text") or "").strip()
        filename = str(processed_doc.get("filename") or "document.pdf")
        return [
            prompt,
            f"Filename: {filename}\n\nDocument text:\n{extracted_text}",
        ]

    if input_mode == "scanned_pdf":
        pdf_bytes = _read_binary_file(processed_doc.get("filepath"), "scanned PDF")
        mime_type = str(processed_doc.get("mime_type") or "application/pdf")
        pdf_part = types_module.Part.from_bytes(data=pdf_bytes, mime_type=mime_type)
        return [pdf_part, prompt]

    # input_mode == "image"
    image_bytes = processed_doc.get("image_bytes")
    if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
        image_bytes = _read_binary_file(processed_doc.get("filepath"), "image")
    else:
        image_bytes = bytes(image_bytes)

    mime_type = str(processed_doc.get("mime_type") or "image/jpeg")
    image_part = types_module.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return [image_part, prompt]


def _build_extraction_schema(types_module: Any) -> Any:
    """Build a Gemini-compatible ``types.Schema`` supporting nullable fields."""
    nullable_str = lambda desc: types_module.Schema(  # noqa: E731
        type=types_module.Type.STRING,
        description=desc,
        nullable=True,
    )

    medicine_schema = types_module.Schema(
        type=types_module.Type.OBJECT,
        properties={
            "name": nullable_str("Medicine name as written, or null."),
            "strength": nullable_str("Medicine strength (e.g., 500 mg), or null."),
            "dosage": nullable_str("Dosage per administration (e.g., 1 tablet), or null."),
            "frequency": nullable_str("Frequency of administration, or null."),
            "timing": nullable_str("Timing instruction (e.g., after meals), or null."),
            "duration": nullable_str("Duration of therapy (e.g., 5 days), or null."),
        },
        required=["name", "strength", "dosage", "frequency", "timing", "duration"],
    )

    test_schema = types_module.Schema(
        type=types_module.Type.OBJECT,
        properties={
            "name": nullable_str("Lab test name as written, or null."),
            "result": nullable_str("Measured result value as written, or null."),
            "unit": nullable_str("Unit of measurement, or null."),
            "reference_range": nullable_str("Printed reference range, or null."),
            "date": nullable_str("Test or report date if available, or null."),
        },
        required=["name", "result", "unit", "reference_range", "date"],
    )

    return types_module.Schema(
        type=types_module.Type.OBJECT,
        properties={
            "document_type": types_module.Schema(
                type=types_module.Type.STRING,
                enum=["prescription", "lab_report", "unknown"],
                description="Classification of the uploaded medical document.",
            ),
            "summary": types_module.Schema(
                type=types_module.Type.STRING,
                description="Brief factual summary without diagnosis or clinical interpretation.",
            ),
            "medicines": types_module.Schema(
                type=types_module.Type.ARRAY,
                items=medicine_schema,
                description="Extracted prescription medications, or empty array.",
            ),
            "tests": types_module.Schema(
                type=types_module.Type.ARRAY,
                items=test_schema,
                description="Extracted laboratory test results, or empty array.",
            ),
        },
        required=["document_type", "summary", "medicines", "tests"],
    )


def _get_gemini_client() -> tuple[Any, Any]:
    """Import ``google-genai`` and initialize an authenticated Gemini client."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise AIConfigurationError(
            "The 'google-genai' package is not installed. Install requirements.txt first."
        ) from exc

    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise AIConfigurationError(
            "Missing Gemini API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment."
        )

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        raise AIConfigurationError(f"Failed to initialize Gemini client: {exc}") from exc

    return client, types


def _parse_extraction_response(response: Any) -> DocumentExtractionResult:
    """Validate and parse the Gemini API response into ``DocumentExtractionResult``."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, DocumentExtractionResult):
            return parsed
        if isinstance(parsed, BaseModel):
            return DocumentExtractionResult.model_validate(parsed.model_dump())
        if isinstance(parsed, dict):
            return DocumentExtractionResult.model_validate(parsed)

    raw_text = getattr(response, "text", None)
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise GeminiAPIError("Gemini returned an empty response.")

    cleaned_text = raw_text.strip()
    # Strip markdown code fences if a model ignores MIME constraints.
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned_text, flags=re.DOTALL)
    if fence_match:
        cleaned_text = fence_match.group(1).strip()

    try:
        payload = json.loads(cleaned_text)
        return DocumentExtractionResult.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise GeminiAPIError(f"Gemini returned invalid structured JSON: {exc}") from exc


def extract_document_with_gemini(
    processed_doc: dict[str, Any],
    language: str = DEFAULT_LANGUAGE,
    *,
    client: Any | None = None,
    types_module: Any | None = None,
    model: str | None = None,
) -> tuple[DocumentExtractionResult, InputMode]:
    """Run Stage 1 Gemini classification and extraction on ``processed_doc``.

    Args:
        processed_doc: Dictionary returned by ``backend.document_processor.process_uploaded_file``.
        language: Desired language for the brief factual summary.
        client: Optional pre-configured ``google.genai.Client`` (useful for testing).
        types_module: Optional ``google.genai.types`` module override for testing.
        model: Optional Gemini model name; defaults to ``GEMINI_MODEL`` env var or
            ``gemini-3.8-flash``.

    Returns:
        Tuple of ``(DocumentExtractionResult, input_mode)``.
    """
    input_mode = detect_input_mode(processed_doc)

    if client is None or types_module is None:
        resolved_client, resolved_types = _get_gemini_client()
        client = client or resolved_client
        types_module = types_module or resolved_types

    target_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    contents = _build_gemini_contents(
        processed_doc=processed_doc,
        input_mode=input_mode,
        language=language,
        types_module=types_module,
    )
    schema = _build_extraction_schema(types_module)

    config = types_module.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=schema,
    )

    try:
        response = client.models.generate_content(
            model=target_model,
            contents=contents,
            config=config,
        )
    except AIProcessorError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GeminiAPIError(f"Gemini API request failed: {exc}") from exc

    extraction = _parse_extraction_response(response)
    return extraction, input_mode


def _build_stage1_response(
    extraction: DocumentExtractionResult,
    input_mode: InputMode,
    language: str,
) -> dict[str, Any]:
    """Convert a validated extraction result into the standard pipeline dictionary."""
    payload = MedTranslateStage1Response(
        document_type=extraction.document_type,
        summary=extraction.summary,
        medicines=extraction.medicines,
        tests=extraction.tests,
        questions_for_doctor=[],
        disclaimer=DEFAULT_DISCLAIMER,
        stage="extraction",
        input_mode=input_mode,
        language=language,
        error=None,
    )
    return payload.model_dump()


def _build_error_fallback_response(
    error_message: str,
    input_mode: InputMode | None,
    language: str,
) -> dict[str, Any]:
    """Return a safe, predictable response when AI extraction fails."""
    payload = MedTranslateStage1Response(
        document_type="unknown",
        summary=(
            "Unable to complete automated document extraction at this time. "
            "Please verify the service configuration or try uploading a clearer document."
        ),
        medicines=[],
        tests=[],
        questions_for_doctor=[],
        disclaimer=DEFAULT_DISCLAIMER,
        stage="extraction",
        input_mode=input_mode,
        language=language,
        error=error_message,
    )
    return payload.model_dump()


def run_medtranslate_pipeline(
    processed_doc: dict[str, Any],
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Public Stage 1 entry point invoked by ``app.py``.

    Classifies the uploaded document (``prescription``, ``lab_report``, or ``unknown``)
    and extracts explicitly present fields into a predictable dictionary matching
    MedTranslate's response contract.
    """
    normalized_language = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    detected_mode: InputMode | None = None

    try:
        detected_mode = detect_input_mode(processed_doc)
        extraction, detected_mode = extract_document_with_gemini(
            processed_doc=processed_doc,
            language=normalized_language,
        )
        return _build_stage1_response(
            extraction=extraction,
            input_mode=detected_mode,
            language=normalized_language,
        )
    except AIConfigurationError as exc:
        logger.error("AI processor configuration error: %s", exc)
        return _build_error_fallback_response(
            error_message=str(exc),
            input_mode=detected_mode,
            language=normalized_language,
        )
    except DocumentInputError as exc:
        logger.warning("Invalid document input passed to AI pipeline: %s", exc)
        return _build_error_fallback_response(
            error_message=str(exc),
            input_mode=detected_mode,
            language=normalized_language,
        )
    except GeminiAPIError as exc:
        logger.error("Gemini extraction error: %s", exc)
        return _build_error_fallback_response(
            error_message="The AI document extraction service encountered an error.",
            input_mode=detected_mode,
            language=normalized_language,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in run_medtranslate_pipeline: %s", exc)
        return _build_error_fallback_response(
            error_message="An unexpected error occurred during AI document extraction.",
            input_mode=detected_mode,
            language=normalized_language,
        )
