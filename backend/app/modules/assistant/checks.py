"""Safety gate and answer checks of the assistant: pure functions, no database, no AI.

Two jobs:
1. Decide from word lists (`safety/words_v1.toml`) whether a text is an emergency, a bypass
   request, a high-voltage topic or a wiring topic. Code decides first, the AI comes second.
2. Check a generated answer against the handbook sections it cites: known sources, every number
   in the body of a cited section (only references like "p. 63" or "step 2" are exempt), every
   part number in a cited section, and no unsafe or wiring words in generated text.
"""

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.modules.assistant.schemas import AnswerDraft, StepDraft, WarningDraft

WORDS_FILE = Path(__file__).resolve().parents[3] / "safety" / "words_v1.toml"

# Safety levels, from least to most strict.
_LEVEL_RANK = {"normal": 0, "high_voltage": 1, "emergency": 2}

# A part of at least this many characters matches anywhere (Dutch compounds).
_MIN_ANYWHERE_LENGTH = 6

# Characters that are dropped before matching because they are invisible or only decorate a
# letter: format (Cf), nonspacing and enclosing marks (Mn, Me), private use (Co), unassigned (Cn).
_HIDDEN_CATEGORIES = {"Cf", "Mn", "Me", "Co", "Cn"}
# Blank "filler" letters that draw nothing (Hangul fillers and the braille blank).
_FILLERS = {"\u115f", "\u1160", "\u3164", "\uffa0", "\u2800"}
# The only non-Latin letters allowed: Greek mu and omega, used in units.
_UNIT_LETTERS = {"\u03bc", "\u03c9"}

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
# A token that starts with letters and holds a digit (AB-471, M8) is checked as a part number.
# The lookbehind keeps "A2" out of "4711-A2": that one is a number, not a token.
_CODE_RE = re.compile(r"(?<![\w/-])[A-Za-z][A-Za-z0-9/-]*")
# Reference patterns (case ignored). A unit after one ("p. 63 Nm", "S1 Nm") makes it a value.
_UNITS = r"(?:(?:nm|n\W?m|kg|mm|cm|km|bar|psi|rpm|hz|ah|kwh|wh|kw|kv|ma|v|w)(?![a-z])|[\u00b0%])"
_NO_UNIT = rf"(?![.,]?\d)(?!\s*{_UNITS})"  # after a reference number: no digit, no unit
_SEP = r"\s*(?:[-\u2010-\u2015\u2212,]|to|tot|en|et|and)\s*"  # "63-64", "63, 64", "63 en 64"
_SOURCE_REF_RE = re.compile(rf"\((S\d{{1,2}})\)|\b(S\d{{1,2}})\b(?!\s*(?:\d|{_UNITS}))", re.I)
_PAGE_RE = re.compile(
    r"\b(?:pp?\.|pages?|pag\.|pagina(?:['\u2019]?s)?|blz\.|bladzijde)\s*(\d+)"
    rf"(?:{_SEP}(\d+))?+{_NO_UNIT}",
    re.I,
)
_SECTION_RE = re.compile(
    r"\b(section|sectie|paragraaf|paragraphe|hoofdstuk|chapter|chapitre|see|zie|voir)"
    rf"\s*(\d+(?:\.\d+)+){_NO_UNIT}",
    re.I,
)
_STEP_RE = re.compile(rf"\b(?:steps?|stap(?:pen)?|etapes?)\s*(\d+){_NO_UNIT}", re.I)


@dataclass(frozen=True)
class SourceText:
    """One cited section: text (`body`), number, heading path and pages (`label`), page range."""

    body: str
    label: str
    page_start: int = 0
    page_end: int = 0

    def full(self) -> str:
        """Label and body together, for the checks that do not care where the text comes from."""
        return f"{self.label}\n{self.body}"


def load_words() -> dict[str, list[str]]:
    """Read the word lists once: list name -> all entries of nl, fr and en together."""
    with WORDS_FILE.open("rb") as file:
        data = tomllib.load(file)
    return {
        name: [e for entries in langs.values() for e in entries] for name, langs in data.items()
    }


WORDS = load_words()


def _is_hidden(char: str) -> bool:
    """True for a character that is invisible or only decorates another (see _HIDDEN_CATEGORIES)."""
    return (
        unicodedata.category(char) in _HIDDEN_CATEGORIES
        or unicodedata.combining(char) != 0
        or char in _FILLERS
    )


def _clean(text: str, control_as_space: bool, punctuation_as_space: bool) -> str:
    """Normalise a text for matching.

    NFKD, drop hidden characters (zero-width, soft hyphen, marks, fillers, private use), handle
    control characters (a space, or removed unless they are whitespace) and punctuation or
    symbols (a space, or removed), drop accents, casefold, and turn any whitespace into one space.
    """
    out: list[str] = []
    for char in unicodedata.normalize("NFKD", text):
        category = unicodedata.category(char)
        if _is_hidden(char):
            continue
        if category in ("Cc", "Cs"):
            if control_as_space or char.isspace():
                out.append(" ")
        elif category[0] in "PS":  # punctuation, dashes and symbols
            if punctuation_as_space:
                out.append(" ")
        else:
            out.append(char)
    return " ".join("".join(out).casefold().split())


def _variants(text: str) -> list[str]:
    """The readings of a text: control characters and punctuation each as a space or removed.

    "bat.tery" is only found when punctuation is removed, "high.voltage" only when it is a space.
    """
    return [_clean(text, c, p) for c in (True, False) for p in (True, False)]


def _part_pattern(part: str) -> re.Pattern[str]:
    """Regex for one part of an entry.

    "word*" is a word start, "*word" a word end, a long part matches anywhere, a short part only
    as a whole word.
    """
    word = re.escape(_clean(part.strip("*"), True, True))
    if part.endswith("*"):
        return re.compile(rf"(?<!\w){word}\w*")
    if part.startswith("*"):
        return re.compile(rf"\w*{word}(?!\w)")
    if len(word) >= _MIN_ANYWHERE_LENGTH:
        return re.compile(word)
    return re.compile(rf"(?<!\w){word}(?!\w)")


def _compile(entries: list[str]) -> list[list[re.Pattern[str]]]:
    """Turn entries into patterns; an entry with '+' becomes several parts that must all match."""
    compiled = []
    for entry in entries:
        parts = [p for p in entry.split("+") if _clean(p.strip("*"), True, True)]
        compiled.append([_part_pattern(p) for p in parts])
    return compiled


_PATTERNS = {name: _compile(entries) for name, entries in WORDS.items() if name != "ignore"}

# Whole words of the [ignore] list. They are blanked out before matching, so "accurate" cannot
# match "accu*".
_IGNORED = "|".join(re.escape(_clean(word, True, True)) for word in WORDS["ignore"])
_IGNORE_RE = re.compile(rf"(?<!\w)(?:{_IGNORED})(?!\w)")


def _matches(list_name: str, text: str) -> bool:
    """True when any entry of the list matches in any reading of the text (stricter wins)."""
    for variant in _variants(text):
        variant = _IGNORE_RE.sub(" ", variant)
        for parts in _PATTERNS[list_name]:
            if all(pattern.search(variant) for pattern in parts):
                return True
    return False


def _has_foreign_letter(text: str) -> bool:
    """True when the text holds a letter that is not Latin (Cyrillic a, Greek, CJK, ...).

    Questions are Dutch, French or English, so such a letter is a way to hide a word. Greek mu
    and omega are allowed because units use them.
    """
    for char in _clean(text, True, True):
        if char.isascii() or not char.isalpha() or char in _UNIT_LETTERS:
            continue
        if "LATIN" not in unicodedata.name(char, ""):
            return True
    return False


def question_level(text: str) -> str:
    """Safety level of a text: emergency before high_voltage before normal.

    A non-Latin letter makes the text at least high_voltage (see `_has_foreign_letter`).
    """
    if _matches("emergency", text):
        return "emergency"
    if _matches("high_voltage", text) or _has_foreign_letter(text):
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


def _number_key(written: str) -> Decimal | str:
    """The value of one number as written in a text.

    Comma equals dot and trailing zeros do not matter (25,0 == 25). Exactly three digits after
    the separator ("1,500") can be a thousands mark or a decimal, so that one is kept as
    written (with a dot) and never equals another number.
    """
    plain = written.replace(",", ".")
    if re.fullmatch(r"\d+\.\d{3}", plain):
        return plain
    return Decimal(plain).normalize()


def _unhide(text: str) -> str:
    """NFKD form of a text without hidden characters (accents, zero-width marks, fillers)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not _is_hidden(c))


def numbers_in(text: str) -> set[Decimal | str]:
    """All numbers in a text, as `_number_key` values; hidden characters are dropped first."""
    return {_number_key(m) for m in _NUMBER_RE.findall(_unhide(text))}


def _has_part_number(part_number: str, source_text: str) -> bool:
    """True when the source holds exactly this part number; case and spaces do not matter.

    It must be whole: not followed or preceded by a letter, digit, "-" or "/", and not by a dot
    or comma with a digit ("471" is not in "471.25" or "4711-A"). Spaces and tabs may differ,
    a line break may not.
    """
    characters = "".join(part_number.split()).casefold()
    if not characters:
        return False
    body = r"[ \t]*".join(re.escape(char) for char in characters)
    pattern = rf"(?<![\w/-])(?<!\d[.,]){body}(?![\w/-])(?![.,]\d)"
    return re.search(pattern, source_text.casefold()) is not None


def _full(cited: list[SourceText]) -> str:
    """Label and body of all cited sections as one text (a part number never spans a line)."""
    return "\n".join(s.full() for s in cited)


def _codes_in(text: str) -> list[str]:
    """Tokens that start with letters and hold a digit (AB-471, M8)."""
    tokens = (token.rstrip("-/") for token in _CODE_RE.findall(text))
    return [token for token in tokens if any(char.isdigit() for char in token)]


def _strip_refs(text: str, sent_ids: set[str], cited: list[SourceText], step_count: int) -> str:
    """Remove the valid references from a text so they never count as numbers.

    Valid: a sent source id (S1); a page (p. 63) in the page range of a cited section; a section
    (section 4.5.3, "see" needs two dots) in a cited label or body; a step from 1 to the number
    of steps. An invalid reference, or one followed by a unit ("p. 63 Nm"), stays in the text.
    """

    def source(m: re.Match[str]) -> str:
        return " " if (m[1] or m[2]) in sent_ids else m[0]

    def page(m: re.Match[str]) -> str:
        pages = [int(n) for n in m.groups() if n]
        in_range = all(any(s.page_start <= n <= s.page_end for s in cited) for n in pages)
        return " " if in_range else m[0]

    def section(m: re.Match[str]) -> str:
        dots_needed = 2 if m[1].lower() in ("see", "zie", "voir") else 1
        valid = m[2].count(".") >= dots_needed and _has_part_number(m[2], _full(cited))
        return " " if valid else m[0]

    def step(m: re.Match[str]) -> str:
        return " " if 1 <= int(m[1]) <= step_count else m[0]

    for pattern, check in [(_SOURCE_REF_RE, source), (_PAGE_RE, page), (_SECTION_RE, section)]:
        text = pattern.sub(check, text)
    return _STEP_RE.sub(step, text)


def check_answer(answer: AnswerDraft, sources: dict[str, SourceText]) -> list[str]:
    """Check a draft against the sent sections; returns the flags raised, empty when fine.

    `sources` maps "S1" to a section. Every number must be in the body of a cited section (for
    missing_information: of any sent section); the label only proves a reference like "p. 63".
    Flags: unknown_source, number_not_in_source, part_number_not_in_source, unsafe_words,
    wiring_words.
    """
    flags: set[str] = set()
    texts: list[str] = []  # all generated text, for the word checks

    def cited_text(ids: list[str]) -> list[SourceText]:
        """The cited sections that were sent; flag ids that were not."""
        known = [sources[i] for i in ids if i in sources]
        if len(known) != len(ids):
            flags.add("unknown_source")
        return known

    def check_values(text: str, cited: list[SourceText]) -> None:
        """Flag numbers and codes (AB-471, M8) in generated text that the sources lack."""
        plain = _strip_refs(_unhide(text), set(sources), cited, len(answer.steps))
        if not numbers_in(plain) <= {n for s in cited for n in numbers_in(s.body)}:
            flags.add("number_not_in_source")
        if not all(_has_part_number(code, _full(cited)) for code in _codes_in(plain)):
            flags.add("part_number_not_in_source")

    items: list[StepDraft | WarningDraft] = [*answer.steps, *answer.warnings]
    for item in items:
        texts.append(item.text)
        check_values(item.text, cited_text(item.sources))

    for part in answer.parts:
        cited = cited_text([part.source])
        texts.append(part.name)
        check_values(part.name, cited)
        if part.part_number:
            texts.append(part.part_number)
            if not _has_part_number(part.part_number, _full(cited)):
                flags.add("part_number_not_in_source")

    for note in answer.missing_information:
        texts.append(note)
        check_values(note, list(sources.values()))

    for text in texts:
        if question_level(text) != "normal" or is_bypass(text):
            flags.add("unsafe_words")
        if is_wiring(text):
            flags.add("wiring_words")
    return sorted(flags)
