"""Document validation and preprocessing utilities for MedTranslate."""

from __future__ import annotations

import mimetypes
import os
from typing import Any

from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

ALLOWED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
ALLOWED_PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})
ALLOWED_EXTENSIONS: frozenset[str] = ALLOWED_PDF_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS

MAX_FILE_SIZE_MB: int = 10
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

_MIME_TYPE_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class DocumentValidationError(ValueError):
    """Raised when an uploaded document fails validation or integrity checks."""


def get_file_extension(filename: str) -> str:
    """Return the normalized lowercase file extension including the leading dot."""
    if not filename or not isinstance(filename, str):
        return ""
    _, ext = os.path.splitext(filename.strip())
    return ext.lower()


def allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed document or image extension."""
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def validate_file_extension(filename: str) -> str:
    """Validate the file extension and return it in lowercase.

    Raises:
        DocumentValidationError: If the filename is missing or has an unsupported extension.
    """
    ext = get_file_extension(filename)
    if not ext or ext not in ALLOWED_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DocumentValidationError(
            f"Unsupported file type '{ext or 'none'}'. Allowed types: {allowed_list}."
        )
    return ext


def validate_file_size(filepath: str, max_bytes: int = MAX_FILE_SIZE_BYTES) -> int:
    """Verify the file exists, is non-empty, and does not exceed max_bytes."""
    if not os.path.isfile(filepath):
        raise DocumentValidationError("Uploaded file could not be found on disk.")

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise DocumentValidationError("Uploaded file is empty.")
    if file_size > max_bytes:
        raise DocumentValidationError(
            f"File size exceeds the maximum limit of {MAX_FILE_SIZE_MB} MB."
        )
    return file_size


def _extract_pdf_text(filepath: str) -> tuple[str | None, bool, int]:
    """Extract text from a PDF file using pypdf.

    Returns:
        A tuple of (extracted_text, is_scanned, page_count).
    """
    try:
        reader = PdfReader(filepath)
        if reader.is_encrypted:
            raise DocumentValidationError(
                "Encrypted or password-protected PDFs are not supported."
            )

        page_count = len(reader.pages)
        if page_count == 0:
            raise DocumentValidationError("The uploaded PDF contains no pages.")

        page_texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            stripped = text.strip()
            if stripped:
                page_texts.append(stripped)

        combined_text = "\n\n".join(page_texts).strip()
        if not combined_text:
            return None, True, page_count

        return combined_text, False, page_count
    except DocumentValidationError:
        raise
    except (PdfReadError, OSError, ValueError) as exc:
        raise DocumentValidationError(
            f"Invalid or corrupted PDF file: {exc}"
        ) from exc


def _inspect_image(filepath: str) -> tuple[dict[str, Any], bytes]:
    """Verify image integrity with Pillow and return metadata plus raw bytes."""
    try:
        # Verify integrity first (verify() closes/invalidates the image object).
        with Image.open(filepath) as img:
            img.verify()

        # Re-open and force-load pixel data to catch truncated files and read metadata.
        with Image.open(filepath) as img:
            img.load()
            image_metadata = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
            }

        with open(filepath, "rb") as image_file:
            image_bytes = image_file.read()

        return image_metadata, image_bytes
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise DocumentValidationError(
            f"Invalid or corrupted image file: {exc}"
        ) from exc


def process_uploaded_file(filepath: str, filename: str) -> dict[str, Any]:
    """Validate and prepare an uploaded PDF or image for downstream AI analysis.

    Args:
        filepath: Path to the temporary saved file on disk.
        filename: Original sanitized or user-provided filename.

    Returns:
        Standardized dictionary containing:
            - filepath: Path to the file on disk.
            - filename: Original filename.
            - file_type: "pdf" or "image".
            - extracted_text: Extracted text for readable PDFs, otherwise None.
            - is_scanned: True for images or PDFs without extractable text.
        Plus helpful metadata (`mime_type`, `file_size_bytes`, `page_count` or
        `image_metadata` and `image_bytes`) for multimodal/OCR pipelines.

    Raises:
        DocumentValidationError: If extension, size, or file integrity checks fail.
    """
    ext = validate_file_extension(filename)
    file_size = validate_file_size(filepath)
    mime_type = _MIME_TYPE_MAP.get(ext) or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    if ext in ALLOWED_PDF_EXTENSIONS:
        extracted_text, is_scanned, page_count = _extract_pdf_text(filepath)
        return {
            "filepath": filepath,
            "filename": filename,
            "file_type": "pdf",
            "extracted_text": extracted_text,
            "is_scanned": is_scanned,
            "mime_type": mime_type,
            "file_size_bytes": file_size,
            "page_count": page_count,
        }

    image_metadata, image_bytes = _inspect_image(filepath)
    return {
        "filepath": filepath,
        "filename": filename,
        "file_type": "image",
        "extracted_text": None,
        "is_scanned": True,
        "mime_type": mime_type,
        "file_size_bytes": file_size,
        "image_metadata": image_metadata,
        "image_bytes": image_bytes,
    }
