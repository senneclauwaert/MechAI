"""Safety gate and answer checks of the assistant: pure functions, no database, no AI.

Two jobs:
1. Decide from word lists (`safety/words_v1.toml`) whether a text is an emergency, a bypass
   request, a high-voltage topic or a wiring topic. Code decides first, the AI comes second.
2. Check a generated answer against the handbook sections it cites: known sources, every number
   and part number present in the cited text, and no unsafe or wiring words in generated text.
"""

import re
import tomllib
import unicodedata
from decimal import Decimal
from pathlib import Path

from app.modules.assistant.schemas import AnswerDraft, StepDraft, WarningDraft

WORDS_FILE = Path(__file__).resolve().parents[3] / "safety" / "words_v1.toml"

# Safety levels, from least to most strict.
_LEVEL_RANK = {"normal": 0, "high_voltage": 1, "emergency": 2}

# A part of at least this many characters matches anywhere (Dutch compounds).
_MIN_ANYWHERE_LENGTH = 6

# Dash-like characters that become a space (category Pd plus the minus sign).
_MINUS_SIGN = "\u2212"

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# Source references in generated text: S1, (S2). They are not numbers from the handbook.
_SOURCE_REF_RE = re.compile(r"\(?\bS\d{1,2}\b\)?")


def load_words() -> dict[str, list[str]]:
    """Read the word lists once: list name -> all entries of nl, fr and en together."""
    with WORDS_FILE.open("rb") as file:
        data = tomllib.load(file)
    return {
        name: [e for entries in langs.values() for e in entries] for name, langs in data.items()
    }


WORDS = load_words()


def _clean(text: str, control_as_space: bool) -> str:
    """Normalise a text for matching.

    NFKD, drop invisible format characters (zero-width, soft hyphen), handle control characters
    (a space, or removed unless they are whitespace), drop accents, casefold, and turn dashes
    and any whitespace into one space.
    """
    out: list[str] = []
    for char in unicodedata.normalize("NFKD", text):
        category = unicodedata.category(char)
        if category == "Cf" or unicodedata.combining(char):
            continue
        if category in ("Cc", "Cs"):
            if control_as_space or char.isspace():
                out.append(" ")
            continue
        if category == "Pd" or char == _MINUS_SIGN:
            out.append(" ")
        else:
            out.append(char)
    return " ".join("".join(out).casefold().split())


def _variants(text: str) -> list[str]:
    """The two readings of a text: control characters as a space, and removed."""
    return [_clean(text, True), _clean(text, False)]


def _part_pattern(part: str) -> re.Pattern[str]:
    """Regex for one part of an entry: anywhere if long, whole word if short."""
    escaped = re.escape(part)
    if len(part) >= _MIN_ANYWHERE_LENGTH:
        return re.compile(escaped)
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _compile(entries: list[str]) -> list[list[re.Pattern[str]]]:
    """Turn entries into patterns; an entry with '+' becomes several parts that must all match."""
    compiled = []
    for entry in entries:
        parts = [_clean(p, True) for p in entry.split("+")]
        compiled.append([_part_pattern(p) for p in parts if p])
    return compiled


_PATTERNS = {name: _compile(entries) for name, entries in WORDS.items()}


def _matches(list_name: str, text: str) -> bool:
    """True when any entry of the list matches in either reading of the text (stricter wins)."""
    for variant in _variants(text):
        for parts in _PATTERNS[list_name]:
            if all(pattern.search(variant) for pattern in parts):
                return True
    return False


def question_level(text: str) -> str:
    """Safety level of a text: emergency before high_voltage before normal."""
    if _matches("emergency", text):
        return "emergency"
    if _matches("high_voltage", text):
        return "high_voltage"
    return "normal"


def is_bypass(text: str) -> bool:
    """True when the text asks to bypass a protection or open a battery pack."""
    return _matches("bypass", text)


def is_wiring(text: str) -> bool:
    """True when the text is about wiring."""
    return _matches("wiring", text)


def strictest(*levels: str) -> str:
    """The strictest of the given levels ('normal' when none are given)."""
    return max(levels, key=_LEVEL_RANK.__getitem__, default="normal")


def numbers_in(text: str) -> set[Decimal]:
    """All numbers in a text; comma equals dot and trailing zeros do not matter (25,0 == 25)."""
    plain = "".join(
        c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Cf"
    )
    return {Decimal(m.replace(",", ".")).normalize() for m in _NUMBER_RE.findall(plain)}


def _has_part_number(part_number: str, source_text: str) -> bool:
    """True when the source holds exactly this part number; spaces and case do not matter.

    It may not be a piece of a longer one: "471" is not found in "4711" or "471-A".
    """
    characters = "".join(part_number.split()).casefold()
    if not characters:
        return False
    pattern = r"\s*".join(re.escape(char) for char in characters)
    return re.search(rf"(?<![\w-]){pattern}(?![\w-])", source_text.casefold()) is not None


def _strip_refs(text: str) -> str:
    """Remove source references like S1 or (S2) so they never count as numbers."""
    return _SOURCE_REF_RE.sub(" ", text)


def check_answer(answer: AnswerDraft, sources: dict[str, str]) -> list[str]:
    """Check a draft against the sent sections; returns the flags raised, empty when fine.

    `sources` maps "S1" to the section text plus its heading path and page range.
    Flags: unknown_source, number_not_in_source, part_number_not_in_source, unsafe_words,
    wiring_words.
    """
    flags: set[str] = set()
    texts: list[str] = []  # all generated text, for the word checks

    def cited_text(ids: list[str]) -> str:
        """Join the cited sections; flag ids that were not sent."""
        known = [sources[i] for i in ids if i in sources]
        if len(known) != len(ids):
            flags.add("unknown_source")
        return "\n".join(known)

    def check_numbers(text: str, source_text: str) -> None:
        """Flag numbers in the generated text that are missing from the given source text."""
        if not numbers_in(_strip_refs(text)) <= numbers_in(source_text):
            flags.add("number_not_in_source")

    items: list[StepDraft | WarningDraft] = [*answer.steps, *answer.warnings]
    for item in items:
        texts.append(item.text)
        check_numbers(item.text, cited_text(item.sources))

    for part in answer.parts:
        source_text = cited_text([part.source])
        texts.append(part.name)
        check_numbers(part.name, source_text)
        if part.part_number:
            texts.append(part.part_number)
            if not _has_part_number(part.part_number, source_text):
                flags.add("part_number_not_in_source")

    all_sources = "\n".join(sources.values())
    for note in answer.missing_information:
        texts.append(note)
        check_numbers(note, all_sources)

    for text in texts:
        if question_level(text) != "normal" or is_bypass(text):
            flags.add("unsafe_words")
        if is_wiring(text):
            flags.add("wiring_words")
    return sorted(flags)
