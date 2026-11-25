"""Flask backend and integration layer for MedTranslate."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from backend.document_processor import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    DocumentValidationError,
    get_file_extension,
    process_uploaded_file,
    validate_file_extension,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("medtranslate.app")

UPLOAD_FOLDER = "uploads"
DEFAULT_DISCLAIMER = (
    "MedTranslate provides educational explanations of your medical documents "
    "and is not a substitute for professional medical advice or diagnosis."
)


def _build_mock_response(processed_doc: dict[str, Any], language: str) -> dict[str, Any]:
    """Return a mock structured JSON response matching the shared team contract.

    Used automatically when Member 3's `backend.ai_processor` module is not yet
    implemented or raises ImportError / NotImplementedError.
    """
    filename = (processed_doc.get("filename") or "").lower()
    extracted_text = (processed_doc.get("extracted_text") or "").lower()
    combined_hint = f"{filename} {extracted_text}"

    lab_keywords = ("hemoglobin", "lab", "reference range", "cbc", "blood", "g/dl", "test")
    rx_keywords = ("rx", "prescription", "metformin", "tablet", "capsule", "dosage", "twice daily")

    is_lab_hint = any(word in combined_hint for word in lab_keywords)
    is_rx_hint = any(word in combined_hint for word in rx_keywords)

    sample_medicine = {
        "name": "Metformin",
        "strength": "500 mg",
        "dosage": "1 tablet",
        "frequency": "twice daily",
        "timing": "after meals",
        "duration": "5 days",
        "purpose": "Used to help control blood sugar levels.",
        "important_information": ["Take with food to reduce stomach upset."],
    }

    sample_test = {
        "name": "Hemoglobin",
        "result": "11.2",
        "unit": "g/dL",
        "reference_range": "12.0 - 16.0 g/dL",
        "status": "Below Reference Range",
        "what_it_measures": "Measures the oxygen-carrying protein in red blood cells.",
        "explanation": (
            "Your value is slightly below the laboratory reference range. "
            "Discuss this with your doctor."
        ),
    }

    if is_lab_hint and not is_rx_hint:
        document_type = "lab_report"
        summary = (
            f"[Mock {language} Output] This laboratory report includes a blood test measurement "
            "showing hemoglobin slightly below the standard reference range. Please review "
            "these findings with your healthcare provider."
        )
        medicines: list[dict[str, Any]] = []
        tests = [sample_test]
        questions = [
            "What could be causing my hemoglobin level to be slightly below range?",
            "Do I need any follow-up blood tests or dietary changes?",
        ]
    elif is_rx_hint and not is_lab_hint:
        document_type = "prescription"
        summary = (
            f"[Mock {language} Output] This prescription outlines a medication schedule for "
            "Metformin 500 mg to be taken with meals. Follow the prescribed timing and confirm "
            "any ongoing refills with your clinician."
        )
        medicines = [sample_medicine]
        tests = []
        questions = [
            "Should I continue taking Metformin after the 5-day course?",
            "What side effects should I watch out for while taking this medication?",
        ]
    else:
        document_type = "prescription"
        summary = (
            f"[Mock {language} Output] Sample structured medical translation showing how "
            "prescribed medications and laboratory results are summarized in plain language. "
            "Connect backend/ai_processor.py for live Gemini analysis."
        )
        medicines = [sample_medicine]
        tests = [sample_test]
        questions = [
            "What could be causing my hemoglobin level to be slightly below range?",
            "Should I take Metformin after meals for the full duration?",
        ]

    return {
        "document_type": document_type,
        "summary": summary,
        "medicines": medicines,
        "tests": tests,
        "questions_for_doctor": questions,
        "disclaimer": DEFAULT_DISCLAIMER,
    }


def _normalize_response_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the API response always conforms to the agreed team JSON contract."""
    if not isinstance(payload, dict):
        raise ValueError("AI pipeline must return a dictionary.")

    doc_type = payload.get("document_type", "unknown")
    if doc_type not in {"prescription", "lab_report", "unknown"}:
        doc_type = "unknown"

    return {
        "document_type": doc_type,
        "summary": payload.get("summary") or "",
        "medicines": payload.get("medicines") if isinstance(payload.get("medicines"), list) else [],
        "tests": payload.get("tests") if isinstance(payload.get("tests"), list) else [],
        "questions_for_doctor": (
            payload.get("questions_for_doctor")
            if isinstance(payload.get("questions_for_doctor"), list)
            else []
        ),
        "disclaimer": payload.get("disclaimer") or DEFAULT_DISCLAIMER,
    }


def _invoke_ai_pipeline(processed_doc: dict[str, Any], language: str) -> dict[str, Any]:
    """Call Member 3's AI pipeline, falling back to mock JSON if unavailable."""
    try:
        from backend.ai_processor import run_medtranslate_pipeline
    except ImportError:
        logger.info("backend.ai_processor not implemented yet; returning mock JSON response.")
        return _build_mock_response(processed_doc, language)

    try:
        result = run_medtranslate_pipeline(processed_doc, language)
        return _normalize_response_contract(result)
    except NotImplementedError:
        logger.info("run_medtranslate_pipeline raised NotImplementedError; returning mock JSON.")
        return _build_mock_response(processed_doc, language)


def _build_safe_upload_path(upload_folder: str, original_filename: str) -> tuple[str, str]:
    """Create a collision-free temporary path inside uploads/ preserving extension."""
    ext = validate_file_extension(original_filename)
    sanitized = secure_filename(original_filename)
    if not sanitized or get_file_extension(sanitized) != ext:
        sanitized = f"document{ext}"

    unique_name = f"{uuid.uuid4().hex}_{sanitized}"
    return os.path.join(upload_folder, unique_name), sanitized


def create_app() -> Flask:
    """Create and configure the MedTranslate Flask application."""
    app = Flask(__name__)
    CORS(app)

    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_exc: RequestEntityTooLarge):
        return (
            jsonify({"error": f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."}),
            400,
        )

    @app.get("/")
    def index():
        template_path = os.path.join(app.root_path, app.template_folder or "templates", "index.html")
        if os.path.isfile(template_path):
            return render_template("index.html")

        return (
            jsonify(
                {
                    "status": "ok",
                    "service": "MedTranslate Backend",
                    "message": (
                        "templates/index.html is not present yet. "
                        "Use GET /api/health or POST /api/analyze."
                    ),
                }
            ),
            200,
        )

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok", "service": "MedTranslate Backend"}), 200

    @app.post("/api/analyze")
    def analyze_document():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request. Expected form field 'file'."}), 400

        uploaded_file = request.files["file"]
        original_filename = (uploaded_file.filename or "").strip()
        if not original_filename:
            return jsonify({"error": "No file selected for upload."}), 400

        language = (request.form.get("language") or "English").strip() or "English"
        temp_filepath: str | None = None

        try:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            temp_filepath, safe_name = _build_safe_upload_path(
                app.config["UPLOAD_FOLDER"], original_filename
            )
            uploaded_file.save(temp_filepath)

            processed_doc = process_uploaded_file(temp_filepath, safe_name)
            result = _invoke_ai_pipeline(processed_doc, language)
            return jsonify(result), 200

        except DocumentValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error analyzing uploaded document: %s", exc)
            return (
                jsonify({"error": "An internal error occurred while analyzing the document."}),
                500,
            )
        finally:
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError as cleanup_exc:
                    logger.warning(
                        "Failed to remove temporary file %s: %s", temp_filepath, cleanup_exc
                    )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
