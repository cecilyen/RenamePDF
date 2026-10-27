"""Portable bibliographic filename policy for the PDF collection."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from record_matching import PaperMeta


MAX_FILENAME_BYTES = 250
PDF_EXTENSION = ".pdf"
MAX_STEM_BYTES = MAX_FILENAME_BYTES - len(PDF_EXTENSION.encode("utf-8"))
TWO_DIGIT_YEAR_PIVOT = 30
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
WORD_FALLBACK = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
STOPWORDS = {
    "the",
    "and",
    "with",
    "for",
    "from",
    "into",
    "onto",
    "upon",
    "over",
    "under",
    "between",
    "among",
    "about",
    "of",
    "in",
    "on",
    "at",
    "by",
    "as",
    "to",
    "an",
    "a",
    "or",
    "nor",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "this",
    "that",
    "these",
    "those",
    "using",
    "via",
}


def strip_diacritics(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value or "")
        if not unicodedata.combining(character)
    )


def name_token(last_name: str) -> str:
    return "".join(
        part if part.isupper() else part.capitalize()
        for part in re.sub(
            r"[^A-Za-z0-9]+",
            " ",
            strip_diacritics(last_name),
        ).split()
        if part
    )


def first_initial_ascii(given: str) -> str:
    match = re.search(r"[A-Za-z0-9]", strip_diacritics(given))
    return match.group(0).upper() if match else ""


def pascal_case_token(phrase: str) -> str:
    return "".join(
        part
        if part.isupper() or any(character.isupper() for character in part[1:])
        else part.capitalize()
        for part in re.findall(
            r"[A-Za-z0-9]+",
            strip_diacritics(phrase or "").strip(),
        )
    )


def short_year(year: int) -> str:
    return f"{year % 100:02d}"


def resolve_two_digit_year(value: int) -> int:
    """Resolve collection year tokens across the supported 1930-2029 span."""
    if not 0 <= value <= 99:
        raise ValueError("two-digit year must be between 0 and 99")
    if value < TWO_DIGIT_YEAR_PIVOT:
        return 2000 + value
    return 1900 + value


def build_keywords(meta: PaperMeta, maximum: int = 8) -> list[str]:
    if meta.author_keywords:
        return [
            pascal_case_token(keyword)
            for keyword in meta.author_keywords[:maximum]
            if keyword
        ]
    if meta.mesh_keywords:
        return [
            pascal_case_token(keyword)
            for keyword in meta.mesh_keywords[:maximum]
            if keyword
        ]

    unique: list[str] = []
    for word in WORD_FALLBACK.findall(meta.title):
        normalized = word.lower()
        if normalized not in STOPWORDS and normalized not in unique:
            unique.append(normalized)
    unique.sort(key=len, reverse=True)
    return [pascal_case_token(keyword) for keyword in unique[:maximum]]


def cap_keywords_by_byte_length(
    base_byte_length: int,
    keywords: list[str],
) -> list[str]:
    accepted: list[str] = []
    running = base_byte_length
    for keyword in keywords:
        keyword_bytes = len(keyword.encode("utf-8"))
        if running + 1 + keyword_bytes > MAX_STEM_BYTES:
            break
        accepted.append(keyword)
        running += 1 + keyword_bytes
    return accepted


def filesystem_safe_pdf_name(stem: str) -> str:
    stem = strip_diacritics(stem or "")
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip(" ._")
    if not stem:
        stem = "Unknown"
    if stem.upper() in WINDOWS_RESERVED_NAMES:
        stem = f"{stem}_paper"

    encoded = stem.encode("utf-8")
    if len(encoded) > MAX_STEM_BYTES:
        stem = encoded[:MAX_STEM_BYTES].decode(
            "utf-8",
            errors="ignore",
        ).rstrip(" ._")
    return stem + PDF_EXTENSION


def disambiguate_pdf_name(filename: str, identity: str) -> str:
    """Append a stable opaque suffix while preserving the filename byte cap."""
    identity = str(identity or "").strip().casefold()
    if not identity:
        raise ValueError("filename disambiguation requires an identity")
    safe_name = filesystem_safe_pdf_name(
        filename[:-len(PDF_EXTENSION)]
        if filename.casefold().endswith(PDF_EXTENSION)
        else filename
    )
    stem = safe_name[:-len(PDF_EXTENSION)]
    suffix = f"_doi{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:10]}"
    limit = MAX_STEM_BYTES - len(suffix.encode("utf-8"))
    stem = stem.encode("utf-8")[:limit].decode(
        "utf-8",
        errors="ignore",
    ).rstrip(" ._")
    return stem + suffix + PDF_EXTENSION


def make_filename(meta: PaperMeta, *, is_arxiv: bool = False) -> str:
    first_family = name_token(meta.first_author.family)
    first_family_normalized = first_family.lower()
    if (
        meta.corresponding_author
        and name_token(meta.corresponding_author.family).lower()
        != first_family_normalized
    ):
        second = meta.corresponding_author
    elif (
        meta.authors
        and name_token(meta.authors[-1].family).lower()
        != first_family_normalized
    ):
        second = meta.authors[-1]
    else:
        second = None

    base = f"{first_family}{first_initial_ascii(meta.first_author.given)}_"
    if second:
        base += f"{name_token(second.family)}_"
    arxiv_record = str(meta.doi or "").casefold().startswith(
        "10.48550/arxiv."
    )
    publication = "arXiv" if is_arxiv or arxiv_record else (
        meta.journal_abbrev or "Journal"
    )
    base += f"{pascal_case_token(publication)}{short_year(meta.year)}"

    keywords = cap_keywords_by_byte_length(
        len(base.encode("utf-8")),
        build_keywords(meta),
    )
    name = "_".join(part for part in [base, *keywords] if part)
    return filesystem_safe_pdf_name(name)
