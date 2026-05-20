"""
skill_extractor.py
------------------
Extracts technical and soft skills from resume/JD text using:
1. A curated, expandable skills taxonomy (exact + alias matching)
2. spaCy NER for named entity recognition
3. Noun-phrase chunking for unseen skill detection

The taxonomy is intentionally broad and covers ML, data, web, cloud, and soft skills.
"""

import re
import logging
from typing import Optional
import spacy
from src.taxonomies.skills_taxonomy import SKILLS_TAXONOMY

logger = logging.getLogger(__name__)


# Compile regex patterns for each skill
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {}

def _get_patterns():
    global _COMPILED_PATTERNS
    if not _COMPILED_PATTERNS:
        for skill, aliases in SKILLS_TAXONOMY.items():
            patterns = []
            for alias in aliases:
                # If alias already has regex special chars, use as-is; else word-boundary wrap
                if any(c in alias for c in r"[]()\\^$.*+?{}|"):
                    pat = re.compile(alias, re.IGNORECASE)
                else:
                    pat = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                patterns.append(pat)
            _COMPILED_PATTERNS[skill] = patterns
    return _COMPILED_PATTERNS


def extract_skills_from_text(text: str) -> set[str]:
    """
    Extract canonical skill names from text using taxonomy matching.
    Returns a set of matched canonical skill names.
    """
    if not text:
        return set()

    found = set()
    patterns = _get_patterns()

    for skill, pats in patterns.items():
        for pat in pats:
            if pat.search(text):
                found.add(skill)
                break  # No need to check other aliases for same skill

    return found


def extract_skills_with_spacy(text: str, nlp=None) -> set[str]:
    """
    Supplementary skill extraction using spaCy noun chunks.
    Returns potential skill phrases not captured by taxonomy.
    """
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return set()

    doc = nlp(text[:50000])
    extra_skills = set()

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        # Keep only short, likely-skill phrases (2-4 words)
        words = chunk_text.split()
        if 1 < len(words) <= 4:
            # Filter out obvious non-skills
            if not any(w in {"the", "a", "an", "my", "our", "their", "i", "we"} for w in words):
                extra_skills.add(chunk_text)

    return extra_skills

def extract_candidate_name(text: str) -> Optional[str]:
    """
    Extract the candidate's full name from resume text.

    Strategy (in priority order):
    1. First non-empty line that looks like a name (2-4 capitalized words, no digits)
    2. Pattern: "Name: John Doe" or "Applicant: John Doe"
    3. Line immediately before an email address
    4. Falls back to None if no confident match found
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    # Pattern: explicit "Name: ..." label
    label_pat = re.compile(
        r"(?:name|applicant|candidate)\s*[:\-]\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})",
        re.IGNORECASE
    )
    for line in lines[:15]:
        m = label_pat.search(line)
        if m:
            return m.group(1).strip()

    # Name-like line: 2–4 words, each capitalized, no digits, not a section header keyword
    SECTION_KEYWORDS = {
        "summary", "objective", "experience", "education", "skills", "profile",
        "contact", "references", "projects", "certifications", "publications",
        "awards", "languages", "interests", "about", "career", "professional",
        "curriculum", "vitae", "resume", "portfolio"
    }
    name_line_pat = re.compile(r"^[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z''\-]+){1,3}$")
    for line in lines[:8]:  # Name is almost always in the first 8 lines
        # Skip lines with digits (phone, address, date) or email symbols
        if re.search(r"[\d@|/\\]", line):
            continue
        if line.lower().split()[0] in SECTION_KEYWORDS:
            continue
        if len(line) > 60:  # Too long to be a name
            continue
        if name_line_pat.match(line):
            return line.strip()

    # Fallback: line before email address
    email_pat = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    for i, line in enumerate(lines[:15]):
        if email_pat.search(line) and i > 0:
            prev = lines[i - 1].strip()
            if name_line_pat.match(prev):
                return prev

    return None


def extract_years_of_experience(text: str) -> Optional[float]:
    """
    Extract years of experience from text.
    Handles patterns like '5 years', '3+ years', '2-4 years', etc.
    """
    patterns = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
        r"(\d+)\+?\s*years?\s+(?:in|working)",
        r"experience\s+of\s+(\d+)\+?\s*years?",
        r"(\d+)\s*-\s*(\d+)\s*years?\s+(?:of\s+)?experience",
    ]
    max_years = None
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                # Range: take average
                try:
                    years = (int(match[0]) + int(match[-1])) / 2
                except (ValueError, IndexError):
                    continue
            else:
                try:
                    years = float(match)
                except ValueError:
                    continue
            if max_years is None or years > max_years:
                max_years = years
    return max_years


def extract_education_level(text: str) -> str:
    """
    Detect highest education level mentioned.
    Returns one of: phd, masters, bachelors, associate, diploma, none
    """
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["ph.d", "phd", "doctor of philosophy", "doctorate"]):
        return "phd"
    if any(kw in text_lower for kw in ["master", "m.s.", "m.sc", "mba", "m.eng", "mtech", "m.tech"]):
        return "masters"
    if any(kw in text_lower for kw in ["bachelor", "b.s.", "b.sc", "b.e.", "b.tech", "btech",
                                         "b.a.", "undergraduate", "b.eng"]):
        return "bachelors"
    if any(kw in text_lower for kw in ["associate", "a.s.", "a.a."]):
        return "associate"
    if any(kw in text_lower for kw in ["diploma", "certificate", "certification"]):
        return "diploma"
    return "none"


def extract_contact_info(text: str) -> dict:
    """Extract email and phone from resume text."""
    email_pat = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    phone_pat = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")

    emails = email_pat.findall(text)
    phones = phone_pat.findall(text)

    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
    }


def full_extraction(text: str, nlp=None) -> dict:
    """
    Run complete extraction on a text block.
    Returns all extracted signals in one dict.
    """
    skills = extract_skills_from_text(text)
    years_exp = extract_years_of_experience(text)
    education = extract_education_level(text)
    contact = extract_contact_info(text)
    candidate_name = extract_candidate_name(text)

    return {
        "skills": skills,
        "years_experience": years_exp,
        "education_level": education,
        "contact": contact,
        "skill_count": len(skills),
        "candidate_name": candidate_name,
    }
