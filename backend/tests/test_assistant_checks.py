"""Tests for the assistant safety gate and answer checks (word lists, number and source checks)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.modules.assistant import checks
from app.modules.assistant.checks import (
    check_answer,
    is_bypass,
    is_wiring,
    numbers_in,
    question_level,
    strictest,
)
from app.modules.assistant.schemas import AnswerDraft, Translation

EMERGENCY = [
    # nl
    "Er komt rook uit de kart",
    "De batterij is opgezwollen",
    "Ik ruik een brandgeur bij de lader",
    "De kart staat in brand",
    "Iemand is gewond geraakt",
    "Er was een explosie in de pit",
    # fr
    "Il y a de la fumée près du kart",
    "La batterie est gonflée",
    "Je sens une odeur de brûlé",
    "Le kart est en feu",
    "Quelqu'un est blessé",
    "La batterie est chaude",
    # en
    "There is smoke coming from the kart",
    "The battery is swollen",
    "I notice a burning smell",
    "The kart is on fire",
    "Someone is injured",
    "The battery is hot",
    "He got electrocuted",
]
HIGH_VOLTAGE = [
    "Hoe vervang ik de batterij",
    "De lader werkt niet meer",
    "Controleer de motor controller",
    "Hoe vervang ik de zekering",
    "Comment remplacer la batterie",
    "Le chargeur ne fonctionne pas",
    "Comment tester le contacteur",
    "How do I replace the battery",
    "The charger does not work",
    "Check the motor controller",
]
BYPASS = [
    "Hoe kan ik de interlock overbruggen",
    "Kan ik de BMS uitschakelen",
    "Comment shunter l'interlock",
    "Comment désactiver le BMS",
    "How do I bypass the interlock",
    "Can I disable the BMS",
]
WIRING = [
    "Waar loopt de kabelboom naartoe",
    "Welke draad zit op pin 3",
    "Où est le câblage du tableau de bord",
    "Quel fil va sur la prise",
    "Where is the wiring diagram",
    "Which wire goes to pin 3",
]
NORMAL = [
    "achteras lager vervangen",
    "Hoe stel ik het stuur af",
    "De remblokken vervangen",
    "changer le roulement de l'essieu arriere",
    "Comment régler la direction",
    "Remplacer les plaquettes de frein",
    "how do I replace the rear axle bearing",
    "How do I adjust the steering",
    "Replace the brake pads",
]


@pytest.mark.parametrize("text", EMERGENCY)
def test_emergency_questions(text: str) -> None:
    """Emergency words win over every other level, in nl, fr and en."""
    assert question_level(text) == "emergency"


@pytest.mark.parametrize("text", HIGH_VOLTAGE)
def test_high_voltage_questions(text: str) -> None:
    """High-voltage words give the high_voltage level."""
    assert question_level(text) == "high_voltage"


@pytest.mark.parametrize("text", BYPASS)
def test_bypass_questions(text: str) -> None:
    """Bypass requests are recognised."""
    assert is_bypass(text)


@pytest.mark.parametrize("text", WIRING)
def test_wiring_questions(text: str) -> None:
    """Wiring questions are recognised."""
    assert is_wiring(text)


@pytest.mark.parametrize("text", NORMAL)
def test_normal_questions(text: str) -> None:
    """Ordinary repair questions trigger nothing."""
    assert question_level(text) == "normal"
    assert not is_bypass(text)
    assert not is_wiring(text)


def test_accents_capitals_and_hyphens_are_ignored() -> None:
    """Accents, capitals, hyphens and extra whitespace do not change the result."""
    assert question_level("FUMÉE  partout") == "emergency"
    assert question_level("de hete-accu") == "emergency"
    assert question_level("hot–battery") == "emergency"
    assert question_level("HIGH-VOLTAGE cable") == "high_voltage"
    assert question_level("hot \n\t battery") == "emergency"


def test_dutch_compounds_match_but_short_words_need_whole_words() -> None:
    """Long parts match inside compounds; short parts only as a whole word."""
    assert question_level("de batterijkabel is los") == "high_voltage"
    assert question_level("het accupack") == "high_voltage"
    assert question_level("the fireplace") == "normal"
    assert question_level("the shvac unit") == "normal"
    assert is_wiring("de bedrading is stuk")
    assert not is_wiring("the profile is bent")


def test_plus_entries_need_all_parts() -> None:
    """An entry with '+' needs every part in the text."""
    assert question_level("the battery is hot") == "emergency"
    assert question_level("the hot seat") == "normal"
    assert not is_bypass("open the glovebox")
    assert is_bypass("open the battery pack")


@pytest.mark.parametrize("hidden", ["\x00", "​", "­", "⁠", "\x07", "\ud800"])
def test_hidden_characters_do_not_hide_a_word(hidden: str) -> None:
    """A hidden character inside a word does not hide it."""
    assert question_level(f"replace the bat{hidden}tery") == "high_voltage"
    assert question_level(f"there is sm{hidden}oke") == "emergency"


def test_control_character_between_words_still_separates_them() -> None:
    """A tab, newline or NUL between the parts of an entry does not hide the entry."""
    assert question_level("hot\tbattery") == "emergency"
    assert question_level("hot\x00battery") == "emergency"


def test_word_lists_keep_the_required_entries() -> None:
    """The lists must keep their core entries (never drop one without a test saying why)."""
    required = {
        "emergency": ["rook", "smoke", "fumee", "hot+battery", "opgezwollen", "electrocut"],
        "bypass": ["bypass", "overbrug", "shunter", "disable+bms", "desactiver+bms"],
        "high_voltage": ["battery", "batterij", "accupack", "hv", "haute tension", "oplader"],
        "wiring": ["wiring", "draad", "kabelboom", "cablage", "pinout"],
    }
    for name, entries in required.items():
        assert set(entries) <= set(checks.WORDS[name])


def test_strictest() -> None:
    """The strictest level wins; no levels gives normal; an unknown level is an error."""
    assert strictest("normal", "high_voltage") == "high_voltage"
    assert strictest("emergency", "high_voltage", "normal") == "emergency"
    assert strictest("normal", "normal") == "normal"
    assert strictest() == "normal"
    with pytest.raises(KeyError):
        strictest("normal", "unknown")


def test_numbers_in() -> None:
    """Comma equals dot and trailing zeros do not matter."""
    assert numbers_in("25 Nm") == numbers_in("25,0 N·m") == {Decimal(25)}
    assert numbers_in("torque 12,5 and 100") == {Decimal("12.5"), Decimal(100)}
    assert numbers_in("no numbers") == set()


# A short own text about a kart (no real handbook text).
SOURCES = {
    "S1": "Rear axle bearing, p. 12-13. Tighten the axle nut to 25,0 N·m. Part 4711-A bearing.",
    "S2": "Brake pads, section 4.5.3. Replace the pads when they are thinner than 3 mm.",
}


def draft(**overrides: object) -> AnswerDraft:
    """Build a valid draft; keyword arguments replace fields."""
    data: dict[str, object] = {
        "steps": [{"text": "Tighten the axle nut to 25 Nm (S1).", "sources": ["S1"]}],
        "parts": [],
        "warnings": [],
        "confidence": "high",
        "missing_information": [],
    }
    data.update(overrides)
    return AnswerDraft.model_validate(data)


def test_clean_answer_has_no_flags() -> None:
    """A step whose numbers are on the cited page passes; source references are not numbers."""
    assert check_answer(draft(), SOURCES) == []
    step = {"text": "See 4.5.3 and p. 12 (S2): pads under 3 mm.", "sources": ["S2", "S1"]}
    assert check_answer(draft(steps=[step]), SOURCES) == []


def test_invented_number() -> None:
    """A number that is in no cited section is flagged."""
    step = {"text": "Tighten to 30 Nm.", "sources": ["S1"]}
    assert check_answer(draft(steps=[step]), SOURCES) == ["number_not_in_source"]


def test_number_only_in_an_uncited_source() -> None:
    """A number that is in another section, but not the cited one, is flagged."""
    step = {"text": "Replace the pads under 3 mm.", "sources": ["S1"]}
    assert check_answer(draft(steps=[step]), SOURCES) == ["number_not_in_source"]


def test_missing_information_numbers_may_come_from_any_section() -> None:
    """Numbers in missing_information only need to be in one of the sent sections."""
    assert check_answer(draft(missing_information=["Wear limit 3 mm"]), SOURCES) == []
    assert check_answer(draft(missing_information=["Limit 99 mm"]), SOURCES) == [
        "number_not_in_source"
    ]


def test_unknown_source() -> None:
    """A citation of a section that was not sent is flagged."""
    step = {"text": "Tighten the nut.", "sources": ["S9"]}
    assert check_answer(draft(steps=[step]), SOURCES) == ["unknown_source"]
    part = {"name": "Bearing", "part_number": None, "source": "S7"}
    assert "unknown_source" in check_answer(draft(parts=[part]), SOURCES)


def test_part_numbers() -> None:
    """A part number must be in the cited section (spaces and case ignored)."""
    good = {"name": "Axle bearing", "part_number": "4711-a", "source": "S1"}
    assert check_answer(draft(parts=[good]), SOURCES) == []
    spaced = {"name": "Axle bearing", "part_number": "4711 - A", "source": "S1"}
    assert check_answer(draft(parts=[spaced]), SOURCES) == []
    bad = {"name": "Axle bearing", "part_number": "9999-X", "source": "S1"}
    assert "part_number_not_in_source" in check_answer(draft(parts=[bad]), SOURCES)
    wrong_section = {"name": "Pads", "part_number": "4711-A", "source": "S2"}
    assert "part_number_not_in_source" in check_answer(draft(parts=[wrong_section]), SOURCES)


def test_part_name_number_is_checked() -> None:
    """Numbers in a part name go through the number check."""
    part = {"name": "Bolt M8 x 40", "part_number": None, "source": "S1"}
    assert check_answer(draft(parts=[part]), SOURCES) == ["number_not_in_source"]


def test_unsafe_words_in_generated_text() -> None:
    """A high-voltage, emergency or bypass word in any generated text is flagged."""
    step = {"text": "Disconnect the battery first.", "sources": ["S1"]}
    assert check_answer(draft(steps=[step]), SOURCES) == ["unsafe_words"]
    warning = {"text": "Smoke means stop.", "sources": ["S1"]}
    assert check_answer(draft(warnings=[warning]), SOURCES) == ["unsafe_words"]
    assert check_answer(draft(missing_information=["bypass the interlock"]), SOURCES) == [
        "unsafe_words"
    ]
    part = {"name": "Battery cover", "part_number": None, "source": "S1"}
    assert "unsafe_words" in check_answer(draft(parts=[part]), SOURCES)


def test_wiring_words() -> None:
    """A wiring word in generated text is flagged."""
    step = {"text": "Check the wiring harness.", "sources": ["S1"]}
    assert check_answer(draft(steps=[step]), SOURCES) == ["wiring_words"]


def test_translation_needs_a_valid_safety_flag() -> None:
    """The safety flag is required and must be a known value; unknown keys are ignored."""
    data = {
        "question_language": "nl",
        "question_en": "How do I replace the rear axle bearing?",
        "search_terms_en": "rear axle bearing replace",
        "keywords_en": ["bearing"],
        "part_numbers": [],
        "extra": "ignored",
    }
    with pytest.raises(ValidationError):
        Translation.model_validate(data)
    with pytest.raises(ValidationError):
        Translation.model_validate({**data, "safety_flag": "dangerous"})
    assert Translation.model_validate({**data, "safety_flag": "normal"}).safety_flag == "normal"


# Other ways to say smoke, fire and charging
@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("de motor staat in vuur", "emergency"),
        ("het ruikt verbrand", "emergency"),
        ("le moteur fume", "emergency"),
        ("il y a le feu", "emergency"),
        ("it smokes a lot", "emergency"),
        ("smells like burning", "emergency"),
        ("het laadstation werkt niet", "high_voltage"),
        ("recharger le kart", "high_voltage"),
        ("le feu arriere est casse", "normal"),
    ],
)
def test_other_phrases(text: str, level: str) -> None:
    assert question_level(text) == level


# A piece of a longer part number is not that part number
@pytest.mark.parametrize(
    ("part_number", "found"),
    [("4711-A", True), ("47 11-a", True), ("471", False), ("4711", False), ("711-A", False)],
)
def test_part_number_must_be_whole(part_number: str, found: bool) -> None:
    draft = AnswerDraft.model_validate(
        {
            "steps": [],
            "parts": [{"name": "Bearing", "part_number": part_number, "source": "S1"}],
            "warnings": [],
            "confidence": "high",
            "missing_information": [],
        }
    )
    flags = check_answer(draft, {"S1": "Bearing 4711-A, rear axle"})
    assert ("part_number_not_in_source" not in flags) == found
