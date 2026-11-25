"""DailyMed REST API v2 retrieval module for MedTranslate.

Retrieves structured, non-personalized FDA drug-label information from the
official U.S. National Library of Medicine (NLM) DailyMed RESTful API v2:
- Search endpoint: ``https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json``
- Label XML endpoint: ``https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml``

This module does NOT use Gemini, embeddings, databases, caching, or patient context.
It only fetches, parses, and cleans official FDA label sections for downstream stages.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests

logger = logging.getLogger("medtranslate.medical_retriever")

DAILYMED_API_BASE: str = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
DAILYMED_DRUG_INFO_URL: str = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
DEFAULT_TIMEOUT_SECONDS: float = 15.0
DEFAULT_SEARCH_PAGE_SIZE: int = 10

# Standard HL7 SPL LOINC section codes
_LOINC_INDICATIONS_AND_USAGE: tuple[str, ...] = ("34067-9", "55106-9")
_LOINC_IMPORTANT_SECTIONS: tuple[tuple[str, str], ...] = (
    ("34066-1", "Boxed Warning"),
    ("34070-3", "Contraindications"),
    ("43685-7", "Warnings and Precautions"),
    ("34071-1", "Warnings"),
    ("34084-4", "Adverse Reactions"),
)

_TARGET_LOINC_CODES: frozenset[str] = frozenset(
    _LOINC_INDICATIONS_AND_USAGE + tuple(code for code, _ in _LOINC_IMPORTANT_SECTIONS)
)

# Sentence/block-bounded limits (never slice mid-sentence)
_MAX_PURPOSE_CHARS: int = 1200
_MAX_WARNING_ITEM_CHARS: int = 900
_MAX_IMPORTANT_ITEMS: int = 6


def _empty_result(
    medicine_name: str,
    retrieval_status: str,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Build a standardized fallback dictionary for failed or empty lookups."""
    return {
        "medicine": medicine_name,
        "source": "DailyMed",
        "source_url": source_url,
        "retrieval_status": retrieval_status,
        "purpose_information": None,
        "important_information": [],
    }


def _clean_text(raw_text: str) -> str:
    """Collapse excessive whitespace and fix spacing around punctuation."""
    text = re.sub(r"\s+", " ", raw_text).strip()
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text


def _split_complete_sentences(text: str) -> list[str]:
    """Split cleaned text into complete sentences without breaking abbreviations/decimals."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    # Split on sentence-ending punctuation followed by whitespace and an uppercase/digit/bullet start
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•(])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _bound_by_complete_sentences(blocks: list[str], max_chars: int) -> str | None:
    """Combine blocks up to ``max_chars`` keeping only complete blocks/sentences.

    Never cuts through the middle of a sentence. If even the first sentence
    exceeds ``max_chars``, the full first sentence is preserved intact.
    """
    selected: list[str] = []
    current_len = 0

    for block in blocks:
        cleaned_block = _clean_text(block)
        if not cleaned_block:
            continue

        for sentence in _split_complete_sentences(cleaned_block):
            addition_len = len(sentence) + (1 if selected else 0)
            if selected and (current_len + addition_len) > max_chars:
                return " ".join(selected)
            selected.append(sentence)
            current_len += addition_len

    return " ".join(selected) if selected else None


def _select_best_spl_match(results: list[dict[str, Any]], medicine_name: str) -> dict[str, Any] | None:
    """Choose the most relevant SPL entry from DailyMed search results."""
    valid_entries = [
        item
        for item in results
        if isinstance(item, dict) and isinstance(item.get("setid"), str) and item["setid"].strip()
    ]
    if not valid_entries:
        return None

    target = medicine_name.strip().lower()

    def _score(entry: dict[str, Any]) -> tuple[int, int]:
        title = str(entry.get("title") or "").strip().lower()
        if title.startswith(target):
            rank = 0
        elif target in title:
            rank = 1
        else:
            rank = 2
        return (rank, len(title))

    return min(valid_entries, key=_score)


def _local_tag(tag: str) -> str:
    """Return the local XML tag name without namespace prefix."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_section_loinc_code(section_elem: ET.Element) -> str | None:
    """Return the LOINC code of an HL7 SPL <section> element, if present."""
    for child in section_elem:
        if _local_tag(child.tag) == "code":
            return child.attrib.get("code")
    return None


def _get_section_title(section_elem: ET.Element) -> str:
    """Return the cleaned <title> text of an HL7 SPL <section> element."""
    for child in section_elem:
        if _local_tag(child.tag) == "title":
            return _clean_text("".join(child.itertext()))
    return ""


def _inline_text(elem: ET.Element) -> str:
    """Extract inline text from an XML element, preserving word boundaries."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        tag = _local_tag(child.tag)
        if tag == "br":
            parts.append(" ")
        else:
            child_text = _inline_text(child)
            if child_text:
                parts.append(child_text)
        if child.tail:
            parts.append(child.tail)
    return _clean_text(" ".join(parts))


def _ensure_terminal_punctuation(text: str) -> str:
    """Ensure a standalone block ends with terminal punctuation if it has none."""
    if not text:
        return text
    if text[-1] in ".!?:;":
        return text
    return f"{text}."


def _extract_list_blocks(list_elem: ET.Element) -> list[str]:
    """Extract readable items from an HL7 SPL <list> element."""
    items: list[str] = []
    caption_text = ""

    for child in list_elem:
        tag = _local_tag(child.tag)
        if tag == "caption":
            caption_text = _inline_text(child)
        elif tag == "item":
            item_text = _inline_text(child)
            if item_text:
                items.append(item_text)

    if not items:
        return [caption_text] if caption_text else []

    # Format short phrase lists into a readable sentence, or keep full-sentence items separate.
    all_short_phrases = all(len(item) < 60 and "." not in item for item in items)
    if all_short_phrases:
        joined = "; ".join(item.rstrip(";,. ") for item in items)
        joined = _ensure_terminal_punctuation(joined)
        if caption_text:
            prefix = caption_text.rstrip(":. ")
            return [f"{prefix}: {joined}"]
        return [joined]

    formatted_items = [_ensure_terminal_punctuation(item) for item in items]
    if caption_text:
        return [_ensure_terminal_punctuation(caption_text), *formatted_items]
    return formatted_items


def _extract_table_blocks(table_elem: ET.Element) -> list[str]:
    """Extract readable row summaries from simple HL7 SPL <table> elements."""
    rows: list[str] = []
    for elem in table_elem.iter():
        if _local_tag(elem.tag) != "tr":
            continue
        cells: list[str] = []
        for cell in elem:
            if _local_tag(cell.tag) in {"th", "td"}:
                cell_text = _inline_text(cell)
                if cell_text:
                    cells.append(cell_text)
        if cells:
            rows.append(_ensure_terminal_punctuation(" — ".join(cells)))
    return rows


def _extract_text_blocks(text_elem: ET.Element) -> list[str]:
    """Extract structured readable text blocks from an HL7 SPL <text> node."""
    blocks: list[str] = []

    if text_elem.text and text_elem.text.strip():
        blocks.append(_ensure_terminal_punctuation(_clean_text(text_elem.text)))

    for child in text_elem:
        tag = _local_tag(child.tag)
        if tag == "list":
            blocks.extend(_extract_list_blocks(child))
        elif tag == "table":
            blocks.extend(_extract_table_blocks(child))
        elif tag in {"paragraph", "content", "renderMultiMedia"}:
            # Handle nested lists inside paragraphs if present
            nested_lists = [sub for sub in child if _local_tag(sub.tag) == "list"]
            if nested_lists:
                direct_parts: list[str] = []
                if child.text:
                    direct_parts.append(child.text)
                for sub in child:
                    if _local_tag(sub.tag) == "list":
                        if direct_parts:
                            lead = _clean_text(" ".join(direct_parts))
                            if lead:
                                blocks.append(_ensure_terminal_punctuation(lead))
                            direct_parts = []
                        blocks.extend(_extract_list_blocks(sub))
                    else:
                        sub_text = _inline_text(sub)
                        if sub_text:
                            direct_parts.append(sub_text)
                    if sub.tail:
                        direct_parts.append(sub.tail)
                if direct_parts:
                    tail_text = _clean_text(" ".join(direct_parts))
                    if tail_text:
                        blocks.append(_ensure_terminal_punctuation(tail_text))
            else:
                para_text = _inline_text(child)
                if para_text:
                    blocks.append(_ensure_terminal_punctuation(para_text))
        else:
            other_text = _inline_text(child)
            if other_text:
                blocks.append(_ensure_terminal_punctuation(other_text))

        if child.tail and child.tail.strip():
            blocks.append(_ensure_terminal_punctuation(_clean_text(child.tail)))

    return [b for b in blocks if b]


def _collect_section_blocks_non_overlapping(
    section_elem: ET.Element,
    *,
    current_code: str | None = None,
    include_title: bool = False,
) -> list[str]:
    """Collect text blocks from a section and its unnamed/non-target subsections once.

    Prevents parent-child <text> duplication while preserving section order.
    """
    blocks: list[str] = []
    title = _get_section_title(section_elem) if include_title else ""

    own_blocks: list[str] = []
    for child in section_elem:
        tag = _local_tag(child.tag)
        if tag == "text":
            own_blocks.extend(_extract_text_blocks(child))

    if title and own_blocks:
        # Avoid repeating the title if the first block already starts with it.
        first_lower = own_blocks[0].lower()
        title_lower = title.rstrip(":").lower()
        if not first_lower.startswith(title_lower):
            blocks.append(f"{title.rstrip(':')}: {own_blocks[0]}")
            blocks.extend(own_blocks[1:])
        else:
            blocks.extend(own_blocks)
    else:
        blocks.extend(own_blocks)

    for child in section_elem:
        if _local_tag(child.tag) != "component":
            continue
        for sub_section in child:
            if _local_tag(sub_section.tag) != "section":
                continue
            sub_code = _get_section_loinc_code(sub_section)
            # If the subsection has a DIFFERENT target LOINC code,
            # it will be collected under that code rather than duplicated here.
            if sub_code and sub_code in _TARGET_LOINC_CODES and sub_code != current_code:
                continue
            blocks.extend(
                _collect_section_blocks_non_overlapping(
                    sub_section,
                    current_code=current_code,
                    include_title=True,
                )
            )

    return blocks


def _walk_sections(elem: ET.Element, collected: dict[str, list[str]]) -> None:
    """Walk the HL7 SPL tree and gather unique blocks per target LOINC code."""
    for child in elem:
        tag = _local_tag(child.tag)
        if tag == "section":
            loinc_code = _get_section_loinc_code(child)
            if loinc_code and loinc_code in _TARGET_LOINC_CODES:
                blocks = _collect_section_blocks_non_overlapping(
                    child,
                    current_code=loinc_code,
                    include_title=False,
                )
                if blocks:
                    existing = collected.setdefault(loinc_code, [])
                    seen = set(existing)
                    for block in blocks:
                        if block not in seen:
                            existing.append(block)
                            seen.add(block)
                # Still inspect child components ONLY for subsections with a DIFFERENT target LOINC code
                for sub in child:
                    if _local_tag(sub.tag) == "component":
                        _walk_distinct_target_subsections(sub, loinc_code, collected)
                continue
        _walk_sections(child, collected)


def _walk_distinct_target_subsections(
    elem: ET.Element,
    parent_code: str,
    collected: dict[str, list[str]],
) -> None:
    """Collect nested subsections only if they carry a different target LOINC code."""
    for child in elem:
        if _local_tag(child.tag) == "section":
            sub_code = _get_section_loinc_code(child)
            if sub_code and sub_code in _TARGET_LOINC_CODES and sub_code != parent_code:
                blocks = _collect_section_blocks_non_overlapping(child, include_title=False)
                if blocks:
                    existing = collected.setdefault(sub_code, [])
                    seen = set(existing)
                    for block in blocks:
                        if block not in seen:
                            existing.append(block)
                            seen.add(block)
                continue
        _walk_distinct_target_subsections(child, parent_code, collected)


def _dedupe_adjacent_phrases(text: str) -> str:
    """Remove accidental repeated leading header clauses inside a single block."""
    # e.g. "X is indicated for: X is indicated in..." -> keep clean sentence flow
    match = re.match(r"^([^:]{10,80}):\s*\1", text, flags=re.IGNORECASE)
    if match:
        return text[len(match.group(1)) + 1 :].strip()
    return text


def _parse_spl_xml(xml_content: str) -> tuple[str | None, list[str]]:
    """Parse HL7 SPL XML to extract clean, non-duplicated label sections."""
    root = ET.fromstring(xml_content)
    sections_by_loinc: dict[str, list[str]] = {}
    _walk_sections(root, sections_by_loinc)

    purpose_information: str | None = None
    for code in _LOINC_INDICATIONS_AND_USAGE:
        blocks = sections_by_loinc.get(code)
        if blocks:
            cleaned_blocks = [_dedupe_adjacent_phrases(b) for b in blocks]
            purpose_information = _bound_by_complete_sentences(cleaned_blocks, _MAX_PURPOSE_CHARS)
            if purpose_information:
                break

    important_information: list[str] = []
    seen_entries: set[str] = set()

    for code, heading in _LOINC_IMPORTANT_SECTIONS:
        blocks = sections_by_loinc.get(code)
        if not blocks:
            continue

        bounded_text = _bound_by_complete_sentences(blocks, _MAX_WARNING_ITEM_CHARS)
        if not bounded_text:
            continue

        # Avoid near-duplicate entries across 43685-7 and 34071-1 if both exist
        if bounded_text in seen_entries:
            continue
        seen_entries.add(bounded_text)

        important_information.append(f"{heading}: {bounded_text}")
        if len(important_information) >= _MAX_IMPORTANT_ITEMS:
            break

    return purpose_information, important_information


def retrieve_medicine_info(
    medicine_name: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Retrieve structured drug-label information for a medicine from DailyMed.

    Args:
        medicine_name: Generic or brand medicine name (e.g., ``"Metformin"``).
        timeout: HTTP request timeout in seconds.
        session: Optional ``requests.Session`` for connection reuse or testing.

    Returns:
        A dictionary with keys:
        ``medicine``, ``source``, ``source_url``, ``retrieval_status``,
        ``purpose_information``, and ``important_information``.
    """
    if not isinstance(medicine_name, str) or not medicine_name.strip():
        return _empty_result(medicine_name="", retrieval_status="invalid_input")

    cleaned_name = _clean_text(medicine_name)
    http = session or requests

    search_url = f"{DAILYMED_API_BASE}/spls.json"
    try:
        search_resp = http.get(
            search_url,
            params={"drug_name": cleaned_name, "pagesize": DEFAULT_SEARCH_PAGE_SIZE, "page": 1},
            timeout=timeout,
        )
        search_resp.raise_for_status()
        search_payload = search_resp.json()
    except requests.Timeout:
        logger.warning("DailyMed search timed out for '%s'.", cleaned_name)
        return _empty_result(cleaned_name, retrieval_status="timeout")
    except requests.HTTPError as exc:
        logger.warning("DailyMed search HTTP error for '%s': %s", cleaned_name, exc)
        return _empty_result(cleaned_name, retrieval_status="http_error")
    except requests.RequestException as exc:
        logger.warning("DailyMed search request failed for '%s': %s", cleaned_name, exc)
        return _empty_result(cleaned_name, retrieval_status="request_error")
    except ValueError as exc:
        logger.warning("DailyMed search returned invalid JSON for '%s': %s", cleaned_name, exc)
        return _empty_result(cleaned_name, retrieval_status="malformed_response")

    if not isinstance(search_payload, dict) or not isinstance(search_payload.get("data"), list):
        return _empty_result(cleaned_name, retrieval_status="malformed_response")

    match = _select_best_spl_match(search_payload["data"], cleaned_name)
    if match is None:
        return _empty_result(cleaned_name, retrieval_status="not_found")

    setid = str(match["setid"]).strip()
    source_url = f"{DAILYMED_DRUG_INFO_URL}?setid={setid}"
    label_xml_url = f"{DAILYMED_API_BASE}/spls/{setid}.xml"

    try:
        label_resp = http.get(label_xml_url, timeout=timeout)
        label_resp.raise_for_status()
        xml_text = label_resp.text
    except requests.Timeout:
        logger.warning("DailyMed label fetch timed out for '%s' (%s).", cleaned_name, setid)
        return _empty_result(cleaned_name, retrieval_status="timeout", source_url=source_url)
    except requests.HTTPError as exc:
        logger.warning("DailyMed label HTTP error for '%s' (%s): %s", cleaned_name, setid, exc)
        return _empty_result(cleaned_name, retrieval_status="http_error", source_url=source_url)
    except requests.RequestException as exc:
        logger.warning("DailyMed label request failed for '%s' (%s): %s", cleaned_name, setid, exc)
        return _empty_result(cleaned_name, retrieval_status="request_error", source_url=source_url)

    if not isinstance(xml_text, str) or not xml_text.strip():
        return _empty_result(cleaned_name, retrieval_status="malformed_response", source_url=source_url)

    try:
        purpose_info, important_info = _parse_spl_xml(xml_text)
    except ET.ParseError as exc:
        logger.warning("DailyMed SPL XML parse error for '%s' (%s): %s", cleaned_name, setid, exc)
        return _empty_result(cleaned_name, retrieval_status="malformed_response", source_url=source_url)

    return {
        "medicine": cleaned_name,
        "source": "DailyMed",
        "source_url": source_url,
        "retrieval_status": "success",
        "purpose_information": purpose_info,
        "important_information": important_info,
    }


def run_retriever_tests() -> bool:
    """Run independent DailyMed retrieval checks for sample and nonexistent medicines."""
    test_cases = [
        ("Metformin", "success"),
        ("Cetirizine", "success"),
        ("Pantoprazole", "success"),
        ("NonexistentMedXYZ999", "not_found"),
    ]

    all_passed = True
    with requests.Session() as session:
        for medicine_name, expected_status in test_cases:
            result = retrieve_medicine_info(medicine_name, session=session)
            status = result.get("retrieval_status")
            passed = status == expected_status
            if expected_status == "success":
                has_data = bool(result.get("purpose_information") or result.get("important_information"))
                passed = passed and has_data

            if passed:
                detail = f"status={status}"
                if status == "success":
                    detail += (
                        f", purpose={'yes' if result.get('purpose_information') else 'no'}"
                        f", important_items={len(result.get('important_information') or [])}"
                    )
                print(f"{medicine_name:<22} -> PASS ({detail})")
            else:
                all_passed = False
                print(f"{medicine_name:<22} -> FAIL (expected={expected_status}, got={status})")

    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_retriever_tests() else 1)
