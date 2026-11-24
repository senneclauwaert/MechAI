"""Tests for the assistant safety gate and answer checks (word lists, number and source checks)."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.modules.assistant import checks
from app.modules.assistant.checks import (
    SourceText,
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
        # added after the first review round
        "ignore": ["accurate", "accurately", "accuracy", "accuraat"],
    }
    required["high_voltage"] += ["accu*", "laad*", "*lader", "regelaar", "accumulateur"]
    required["bypass"] += ["omzeil", "bms+uit", "open+battery", "ouverture+batterie"]
    required["emergency"] += ["electric shock", "heet+accu", "rookwolk", "choc electrique"]
    required["wiring"] += ["connector", "stekker", "rode+kabel", "pin+cable"]
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
    "S1": SourceText(
        body="Tighten the axle nut to 25,0 N·m. Part 4711-A bearing.",
        label="Rear axle bearing, p. 12-13",
        page_start=12,
        page_end=13,
    ),
    "S2": SourceText(
        body="Replace the pads when they are thinner than 3 mm.",
        label="Brake pads, section 4.5.3",
    ),
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
    # 40 is not in the section, and neither is the code M8.
    assert check_answer(draft(parts=[part]), SOURCES) == [
        "number_not_in_source",
        "part_number_not_in_source",
    ]


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


def test_translation_needs_valid_flags() -> None:
    """The three flags are required and must be valid; unknown keys are ignored."""
    data = {
        "question_language": "nl",
        "question_en": "How do I replace the rear axle bearing?",
        "search_terms_en": "rear axle bearing replace",
        "keywords_en": ["bearing"],
        "part_numbers": [],
        "extra": "ignored",
    }
    flags = {"safety_flag": "normal", "is_bypass": False, "is_wiring": False}
    with pytest.raises(ValidationError):
        Translation.model_validate(data)
    with pytest.raises(ValidationError):
        Translation.model_validate({**data, **flags, "safety_flag": "dangerous"})
    for name in ("is_bypass", "is_wiring"):
        missing = {key: value for key, value in flags.items() if key != name}
        with pytest.raises(ValidationError):
            Translation.model_validate({**data, **missing})
        with pytest.raises(ValidationError):  # a real boolean only: "false" and 0 are refused
            Translation.model_validate({**data, **flags, name: "false"})
        with pytest.raises(ValidationError):
            Translation.model_validate({**data, **flags, name: 0})
    translation = Translation.model_validate({**data, **flags, "is_bypass": True})
    assert translation.safety_flag == "normal" and translation.is_bypass is True


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
    flags = check_answer(draft, {"S1": src("Bearing 4711-A, rear axle")})
    assert ("part_number_not_in_source" not in flags) == found


def src(body: str, label: str = "", page_start: int = 0, page_end: int = 0) -> SourceText:
    """A cited section with this body text and, when given, this label and page range."""
    return SourceText(body=body, label=label, page_start=page_start, page_end=page_end)


def step_flags(text: str, source_text: str) -> list[str]:
    """The flags of one step that cites a single section S1 holding `source_text` as its body."""
    step = {"text": text, "sources": ["S1"]}
    return check_answer(draft(steps=[step]), {"S1": src(source_text)})


# ---- Hidden characters, look-alikes and punctuation inside a word -----------------------------

# Every character the security review named, plus private use and an unassigned one.
HIDDEN = [
    "\ufe0f",  # variation selector (Mn)
    "\u034f",  # combining grapheme joiner (Mn)
    "\u180b",  # Mongolian free variation selector (Mn)
    "\u17b4",  # Khmer inherent vowel (Mn)
    "\u20dd",  # combining enclosing circle (Me)
    "\u0488",  # combining Cyrillic hundred thousands sign (Me)
    "\u3164",  # Hangul filler
    "\u115f",  # Hangul choseong filler
    "\u1160",  # Hangul jungseong filler
    "\uffa0",  # halfwidth Hangul filler
    "\u2800",  # braille blank
    "\ue000",  # private use (Co)
    "\u0378",  # unassigned (Cn)
]


@pytest.mark.parametrize("hidden", HIDDEN)
def test_more_hidden_characters_do_not_hide_a_word(hidden: str) -> None:
    """A mark, filler, private-use or unassigned character inside a word does not hide it."""
    assert question_level(f"replace the bat{hidden}tery") == "high_voltage"
    assert question_level(f"there is sm{hidden}oke") == "emergency"
    assert is_bypass(f"how to byp{hidden}ass the interlock")
    assert is_wiring(f"which w{hidden}ire goes there")


@pytest.mark.parametrize("hidden", HIDDEN)
def test_hidden_characters_in_a_generated_step_are_flagged(hidden: str) -> None:
    """The same characters inside a generated step raise unsafe_words."""
    assert step_flags(f"Remove the bat{hidden}tery lid.", "Remove the lid.") == ["unsafe_words"]


@pytest.mark.parametrize("text", ["bat.tery", "bat_tery", "high.voltage", "high_voltage", "H.V."])
def test_punctuation_inside_a_word_does_not_hide_it(text: str) -> None:
    """Punctuation and symbols are read both as a space and as nothing."""
    assert question_level(f"replace the {text} part") == "high_voltage"
    assert step_flags(f"Remove the {text} part.", "Remove the part.") == ["unsafe_words"]


def test_punctuation_inside_an_emergency_word() -> None:
    """The same for emergency words."""
    assert question_level("there is sm.oke") == "emergency"
    assert question_level("there is sm_oke") == "emergency"


def test_a_non_latin_letter_makes_a_question_high_voltage() -> None:
    """A Cyrillic, Greek or Chinese letter is a look-alike or foreign: at least high voltage."""
    assert question_level("replace the b\u0430ttery") == "high_voltage"  # Cyrillic a
    assert question_level("check the axle \u03b1") == "high_voltage"  # Greek alpha
    assert question_level("check the axle \u4e2d") == "high_voltage"  # a Chinese letter
    assert question_level("there is sm\u043eke") == "high_voltage"  # Cyrillic o, not emergency
    assert step_flags("Tighten the n\u0443t.", "Tighten the nut.") == ["unsafe_words"]


@pytest.mark.parametrize("text", ["gap of 5 \u03bcm", "gap of 5 \u00b5m", "10 \u03c9", "10 \u2126"])
def test_greek_mu_and_omega_are_allowed_for_units(text: str) -> None:
    """Micro and ohm signs are units, not look-alikes."""
    assert question_level(text) == "normal"
    assert step_flags(f"Keep the gap small, in {text}.", f"Gap {text}.") == []


def test_accents_and_punctuation_in_normal_text_stay_normal() -> None:
    """Dutch and French accents, apostrophes and quotes do not raise the level."""
    assert question_level("Hoe stel ik d'achteras af? éèê ç ï") == "normal"
    assert question_level("L\u2019essieu arrière \u2014 \u00ab régler \u00bb") == "normal"


# ---- High-voltage compounds -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "De accukabel is los",
        "De acculader werkt niet",
        "Vervang de accuklem",
        "De laadkabel is stuk",
        "Controleer de motorregelaar",
        "De snellader klikt",
        "The accumulator is empty",
        "L'accumulateur est vide",
    ],
)
def test_high_voltage_compounds(text: str) -> None:
    """Dutch compounds with accu, laad and lader are high voltage."""
    assert question_level(text) == "high_voltage"


@pytest.mark.parametrize(
    "text",
    ["Adjust the nut accurately", "This is accurate", "Check the accuracy", "Het is accuraat"],
)
def test_words_that_only_look_like_accu_stay_normal(text: str) -> None:
    """The ignore list keeps accurate, accurately, accuracy and accuraat out of accu*."""
    assert question_level(text) == "normal"


def test_a_step_with_a_compound_is_dropped_and_accurately_stays() -> None:
    """A generated step with "accukabel" is flagged; "accurately" is fine."""
    assert step_flags("Koppel de accukabel los.", "Koppel los.") == ["unsafe_words"]
    assert step_flags("Adjust the nut accurately.", "Adjust the nut.") == []


# ---- Wiring -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "De rode kabel gaat op pin 25",
        "Le cable rouge va sur la broche 25",
        "The red cable goes on pin 25",
        "Waar zit de zwarte kabel",
        "De blauwe kabel is los",
        "Which yellow cable is it",
        "Le cable noir est coupe",
        "Welke stekker is dit",
        "Where does this connector go",
        "Ce connecteur est casse",
        "Which pin on this cable",
        "Welke pin zit op de kabel",
        "Which pin on the connector",
    ],
)
def test_more_wiring_questions(text: str) -> None:
    """Cable colours, connectors and pins are wiring."""
    assert is_wiring(text)


@pytest.mark.parametrize(
    "text",
    ["Adjust the brake cable", "De gaskabel afstellen", "Regler le cable de frein", "Fit a pin"],
)
def test_a_plain_cable_is_not_wiring(text: str) -> None:
    """Brake and throttle cables are everyday work."""
    assert not is_wiring(text)


# ---- Bypass and opening a battery -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hoe kan ik de noodstop omzeilen",
        "Can I override the seat switch",
        "How to override the safety",
        "Comment contourner la sécurité",
        "Comment neutraliser la sécurité",
        "Schakel de BMS uit",
        "BMS uitzetten",
        "Kan de bms uit",
        "Disabling the BMS",
        "How to open the battery",
        "Opening the battery pack",
        "Hoe kan ik de accu openen",
        "De accupack open maken",
        "De accupack openen",
        "De batterijpak openen",
        "De accu openmaken",
        "Ouverture de la batterie",
    ],
)
def test_more_bypass_wordings(text: str) -> None:
    """Bypass wordings of the reviews, in nl, fr and en."""
    assert is_bypass(text)


# ---- Emergency --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "De accu is heet",
        "Een hete accu",
        "De accu is warm",
        "Een warme accu",
        "Een warme batterij",
        "I got an electric shock",
        "Ik kreeg een elektrische schok",
        "J'ai pris un choc électrique",
        "Ik heb een schok gekregen",
        "Iemand is gekwetst",
        "De monteur is verwond",
        "The kart is burning",
        "There is a burning kart",
        "La batterie est brûlante",
        "Er is brand in de pit",
        "Er is rookontwikkeling",
        "Er hangt een rookwolk",
    ],
)
def test_more_emergencies(text: str) -> None:
    """Hot battery, electric shock, injury, burning and smoke wordings are emergencies."""
    assert question_level(text) == "emergency"


# ---- Numbers and part numbers -----------------------------------------------------------------


def test_only_sent_source_ids_are_references() -> None:
    """S45 and (S99) are numbers when only S1 and S2 were sent; so is S1 in front of a unit."""
    section = "Rear axle. Tighten the axle nut to 25 Nm."
    # The unknown reference is also a token that starts with a letter, so a second flag is raised.
    for text in ("Tighten to S45 Nm.", "Tighten to (S99) Nm.", "Tighten to S1 Nm.", "S1 25 Nm."):
        assert "number_not_in_source" in step_flags(text, section)
    assert step_flags("Tighten to 25 Nm (S1).", section) == []
    assert step_flags("Tighten to 25 Nm, see S1.", section) == []
    assert step_flags("Tighten to (S1) 25 Nm.", section) == []


def test_a_thousands_form_is_not_a_decimal() -> None:
    """ "1,500" is compared as written: 1.5 does not match it, 1.500 does."""
    section = "Limit 1,500 mm."
    assert step_flags("Limit 1.5 mm.", section) == ["number_not_in_source"]
    assert step_flags("Limit 1500 mm.", section) == ["number_not_in_source"]
    assert step_flags("Limit 1.500 mm.", section) == []
    assert step_flags("Limit 1,500 mm.", section) == []
    assert step_flags("Limit 1.500 mm.", "Limit 1.5 mm.") == ["number_not_in_source"]
    assert numbers_in("1,500") == numbers_in("1.500") == {"1.500"}
    assert numbers_in("25,0") == {Decimal(25)}  # one trailing zero is not a thousands form


def test_hidden_characters_do_not_split_a_number() -> None:
    """A mark between digits does not turn 25 into 2 and 5."""
    assert numbers_in("2\u034f5 Nm") == {Decimal(25)}


# A part number is whole: no letter, digit, "-", "/" or dot and digit next to it, no line break
@pytest.mark.parametrize(
    ("part_number", "source_text", "found"),
    [
        ("471", "Part 471.25 here", False),
        ("25", "Part 471.25 here", False),
        ("471", "Part 471/B here", False),
        ("471", "Part A/471 here", False),
        ("4711", "page 47\n11 bolts", False),
        ("4711", "page 47 11 bolts", True),
        ("4711", "page 47\t11 bolts", True),
        ("4711", "Part 4711, and 4712", True),
        ("4711", "Part 4711.", True),
        ("4711-A", "Part 4711-A.", True),
        ("4711", "Part 4711-A", False),
    ],
)
def test_part_number_is_whole(part_number: str, source_text: str, found: bool) -> None:
    """Part numbers must be whole in the cited section."""
    part = {"name": "Bearing", "part_number": part_number, "source": "S1"}
    flags = check_answer(draft(steps=[], parts=[part]), {"S1": src(source_text)})
    assert ("part_number_not_in_source" not in flags) == found


def test_a_code_in_generated_text_must_be_in_the_cited_section() -> None:
    """A token like AB-471 or M8 must be there as a whole, else part_number_not_in_source."""
    assert step_flags("Use part AB-471.", "Use part CD-471.") == ["part_number_not_in_source"]
    assert step_flags("Use part AB-471.", "Use part AB-471.") == []
    assert step_flags("Use an M8 bolt.", "Use an M8 bolt.") == []
    assert step_flags("Use an M8 bolt.", "Use an M8x40 bolt.") == ["part_number_not_in_source"]
    assert step_flags("Use an M8 bolt.", "Use a bolt.") == [
        "number_not_in_source",
        "part_number_not_in_source",
    ]
    # Only a token that starts with letters counts; the end of 4711-A2 is part of the number.
    assert step_flags("Use bearing 4711-A2.", "Use bearing 4711-A2.") == []
    assert check_answer(
        draft(steps=[], missing_information=["Is AB-471 right?"]), {"S1": src("Part CD-471")}
    ) == ["part_number_not_in_source"]


# ---- Every number must be in the body, whatever follows it (fix list 5, item 1) ---------------

# One example of every unit, as it may be written in an answer.
UNIT_FORMS = [
    "Nm", "NM", "N·m", "N m", "newton metre", "Newton meters", "V", "kV", "volt", "A",
    "mA", "ampere", "Ah", "W", "kW", "Wh", "kWh", "mm", "millimetre", "millimeter", "cm", "m",
    "km", "kg", "g", "bar", "psi", "rpm", "Hz", "°", "°C", "degrees", "graden",
    "degrés", "degres", "%", "percent", "procent", "pour cent",
]  # fmt: skip


def label_only_flags(text: str, body: str = "Tighten the nut.") -> list[str]:
    """The flags of one step when the section number, title and pages are 4.5, 63 and 70."""
    source = src(body, "4.5 Rear axle pages 63-70", 63, 70)
    return check_answer(draft(steps=[{"text": text, "sources": ["S1"]}]), {"S1": source})


@pytest.mark.parametrize("unit", UNIT_FORMS)
def test_a_number_with_any_unit_is_not_taken_from_the_label(unit: str) -> None:
    """Page 63 in the label does not prove "63 <unit>"; the same number in the body does."""
    assert label_only_flags(f"Tighten to 63 {unit}.") == ["number_not_in_source"]
    assert label_only_flags(f"Tighten to 63{unit}.") == ["number_not_in_source"]
    assert label_only_flags(f"Tighten to 63 {unit}.", body=f"Tighten to 63 {unit}.") == []


def test_numbers_with_a_unit_from_the_label_are_caught() -> None:
    """A page number or a section number is not a torque."""
    assert label_only_flags("Tighten to 63 Nm.") == ["number_not_in_source"]
    assert label_only_flags("Tighten to 4.5 Nm.") == ["number_not_in_source"]
    assert label_only_flags("Tighten to 63-70 Nm.") == ["number_not_in_source"]
    assert label_only_flags("Tighten to 25 Nm.", body="Tighten to 25 Nm.") == []
    assert label_only_flags("Tighten to 25,0 N\u00b7m.", body="Tighten to 25 Nm.") == []


# Every spelling of a unit or a range sign: the number is still not in the body
@pytest.mark.parametrize(
    "text",
    [
        "63 Nm", "63 N.m", "63 N-m", "63 N\u22c5m", "63 N\u00b7m", "63 newton-meter",
        "63 newton metre", "63 +/- 5 Nm", "63 \u00b1 5 Nm", "63\u20105 Nm", "63\u20115 Nm",
        "63\u22125 Nm", "63 turns", "63 tot en met 20 Nm", "**63** Nm", "63",
    ],
)  # fmt: skip
def test_a_number_is_caught_whatever_unit_or_sign_follows(text: str) -> None:
    """With 63 only in the label (pages 63-70), every one of these is flagged."""
    assert "number_not_in_source" in label_only_flags(f"Tighten to {text}.", body="Tighten it.")


@pytest.mark.parametrize(
    "text",
    [
        "20-25 Nm", "20 - 25 Nm", "20\u201325 Nm", "20 to 25 Nm", "20 tot 25 Nm", "20 a 25 Nm",
        "20 et 25 Nm", "20 en 25 Nm", "20 and 25 Nm", "20 or 25 Nm", "20 of 25 Nm",
        "20 ou 25 Nm", "20/25 Nm", "20 to 25 NM",
    ],
)  # fmt: skip
def test_both_numbers_of_a_range_must_be_in_the_body(text: str) -> None:
    """In "20-25 Nm" and its other forms, 20 and 25 both need to be in the body."""
    assert label_only_flags(f"Tighten to {text}.", body="Tighten to 20 Nm.") == [
        "number_not_in_source"
    ]
    assert label_only_flags(f"Tighten to {text}.", body="Use 20 and 25 Nm.") == []


def test_references_are_not_numbers() -> None:
    """Source ids, pages, sections and steps are taken out before the numbers are read."""
    steps = [{"text": f"Do step {n}.", "sources": ["S1"]} for n in (1, 2, 3)]
    for text in (
        "See p. 63.", "See page 63.", "Zie pagina 63.", "See pages 63-64.", "Zie pagina 63 en 64.",
        "Voir pp. 63, 64.", "Zie blz. 70.", "Lees section 4.5 first.", "Lees sectie 4.5 first.",
        "See 4.5.3 now.", "Lees paragraaf 4.5.3.", "Lire chapitre 4.5.3.", "Tighten (S1).",
        "Do step 3.", "Doe stap 2.", "Faites etape 1.", "Faites \u00e9tape 1.",
    ):  # fmt: skip
        body = "Tighten it. See 4.5.3 here."
        source = src(body, "4.5 Rear axle pages 63-70", 63, 70)
        assert (
            check_answer(draft(steps=steps + [{"text": text, "sources": ["S1"]}]), {"S1": source})
            == []
        )


def test_a_reference_that_fails_its_own_rule_is_checked_as_a_number() -> None:
    """A page outside the range, a step too high, or a unit after the reference."""
    for text in ("See p. 99.", "Do step 40.", "See p. 63 Nm.", "Do step 3 bar.", "Use S1 Nm."):
        assert "number_not_in_source" in label_only_flags(text), text
    assert "number_not_in_source" in label_only_flags("See pages 63-99.")
    assert "number_not_in_source" in label_only_flags("See see 4.5.", body="Nothing.")
    assert "number_not_in_source" in label_only_flags("See section 9.9.9.")
    assert label_only_flags("See p. 63.") == []


def test_section_references_need_the_right_number_of_dots() -> None:
    """ "section 4.5" is found in the label; "see 4.5" has one dot only, so it is a number."""
    assert label_only_flags("Lees section 4.5 first.") == []
    assert label_only_flags("See 4.5 first.") == ["number_not_in_source"]
    assert label_only_flags("See 4.5.3 for details.", body="See 4.5.3 for details.") == []


def test_the_body_of_any_cited_section_counts() -> None:
    """With two cited sections the number may be in either body, but not in a label."""
    first = src("Tighten to 25 Nm.", "4.5 Axle")
    second = src("Wear limit 3 mm.", "4.6 Pads page 70")
    steps = [{"text": "Use 25 Nm and replace under 3 mm.", "sources": ["S1", "S2"]}]
    assert check_answer(draft(steps=steps), {"S1": first, "S2": second}) == []
    steps = [{"text": "Use 70 Nm.", "sources": ["S1", "S2"]}]
    assert check_answer(draft(steps=steps), {"S1": first, "S2": second}) == ["number_not_in_source"]


# ---- Bypass of safety devices (fix list 4, item 3) --------------------------------------------

DEVICES = [
    "noodstop", "emergency stop", "arret d'urgence", "remsensor", "brake sensor", "seat sensor",
    "capteur",
]  # fmt: skip
SWITCH_OFF_WORDS = [
    "uitschakel", "overbrug", "disable", "disabling", "defeat", "desactiv", "shunt",
]  # fmt: skip


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("word", SWITCH_OFF_WORDS)
def test_every_safety_device_with_every_switch_off_word_is_a_bypass(device: str, word: str) -> None:
    """Each device with each way to switch it off is a bypass, in any order."""
    assert is_bypass(f"{device} {word}")
    assert is_bypass(f"{word} the {device} please")


@pytest.mark.parametrize(
    "text",
    [
        "noodstop uitschakelen",
        "De noodstop overbruggen",
        "Schakel de noodstop uit",
        "schakel de remsensor uit",
        "De remsensor uitschakelen",
        "Disable the brake sensor",
        "Disable the seat sensor",
        "Defeat the emergency stop",
        "Disabling the seat sensor",
        "desactiver l'arret d'urgence",
        "Désactiver l'arrêt d'urgence",
        "Comment shunter le capteur de siège",
        "Désactiver le capteur",
    ],
)
def test_switching_off_a_safety_device_is_a_bypass(text: str) -> None:
    """The questions of the reviews, in nl, fr and en."""
    assert is_bypass(text)


@pytest.mark.parametrize(
    "text", ["De remsensor vervangen", "Replace the brake sensor", "Test the emergency stop"]
)
def test_working_on_a_safety_device_is_not_a_bypass(text: str) -> None:
    """Replacing or testing the device is normal work."""
    assert not is_bypass(text)


# ---- Narrower emergency words (fix list 4, items 5 to 7) --------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "La batterie est gonflée",
        "batterie gonflee",
        "la batterie a gonflé",
        "l'accu a gonflé",
        "le pack gonfle",
        "une cellule gonflee",
        "La batterie va gonfler",
    ],
)
def test_gonfle_is_an_emergency_only_with_a_battery_word(text: str) -> None:
    """A swollen battery is an emergency (fr)."""
    assert question_level(text) == "emergency"


@pytest.mark.parametrize(
    "text",
    ["gonfler les pneus", "pneu gonflé", "pneu degonfle", "Le pneu est gonflé trop fort"],
)
def test_gonfle_alone_is_a_tyre_not_an_emergency(text: str) -> None:
    """Inflating a tyre is normal work; the old list made it an emergency."""
    assert question_level(text) == "normal"


@pytest.mark.parametrize(
    "text",
    [
        "motor verbrand",
        "verbrande accu",
        "de batterij is verbrand",
        "er hangt verbrande rook",
        "ik heb mijn hand verbrand",
        "zijn handen zijn verbrand",
        "ik ben verbrand",
        "hij heeft zijn arm verbrand",
        "verbrand gezicht",
        "mijn vinger is verbrand",
        "verbrand been",
        "het ruikt verbrand",
    ],
)
def test_verbrand_is_an_emergency_with_a_battery_motor_smoke_or_person(text: str) -> None:
    """Burnt battery, motor, smoke or a burnt person (nl) is an emergency."""
    assert question_level(text) == "emergency"


@pytest.mark.parametrize(
    "text",
    [
        "remblokken verbrand",
        "verbrande remblokken",
        "Er is iets verbrand",
        "de koppeling is verbrand",
    ],
)
def test_verbrand_alone_is_not_an_emergency(text: str) -> None:
    """Burnt brake pads are not an emergency; the old list made them one."""
    assert question_level(text) == "normal"


@pytest.mark.parametrize(
    "text",
    [
        "Accueil du kart dans la pit",
        "Il faut accueillir le client",
        "Il accueille les pilotes",
        "De gasregelaar afstellen",
        "De gasregelaars zijn los",
        "De laadruimte is vuil",
        "Alle laadruimtes schoonmaken",
    ],
)
def test_harmless_words_that_start_like_a_safety_word_stay_normal(text: str) -> None:
    """The [ignore] list blanks these whole words, so accu*, laad* and regelaar do not match."""
    assert question_level(text) == "normal"


@pytest.mark.parametrize(
    "text", ["De accukabel in de laadruimte", "De acculader bij de gasregelaar", "L'accumulateur"]
)
def test_a_real_compound_next_to_an_ignored_word_is_still_high_voltage(text: str) -> None:
    """Only whole ignored words are blanked, so a real compound next to one is not lost."""
    assert question_level(text) == "high_voltage"


# ---- A part number "null" and empty translations (fix list 4, items 8 and 9) ------------------


@pytest.mark.parametrize("written", ["null", "NULL", " Null ", "", "   ", None])
def test_a_part_number_null_or_blank_becomes_none(written: str | None) -> None:
    """The model writes "null" for a part without a number; that must not be checked as one."""
    part = {"name": "Bolt", "part_number": written, "source": "S1"}
    answer = draft(steps=[], parts=[part])
    assert answer.parts[0].part_number is None
    assert check_answer(answer, {"S1": src("A bolt.")}) == []


def test_a_real_part_number_is_kept() -> None:
    """Only null and blank are turned into None."""
    part = {"name": "Bearing", "part_number": "4711-A", "source": "S1"}
    answer = draft(steps=[], parts=[part])
    assert answer.parts[0].part_number == "4711-A"
    assert check_answer(answer, {"S1": src("Bearing 4711-A")}) == []


def test_the_answer_prompt_does_not_ask_for_null() -> None:
    """The prompt tells the model to leave the key out, not to write null."""
    from app.modules.assistant import service

    assert "number or null" not in service.ANSWER_PROMPT
    assert "leave `part_number` out" in service.ANSWER_PROMPT


TRANSLATION_DATA = {
    "question_language": "nl",
    "question_en": "How do I replace the rear axle bearing?",
    "search_terms_en": "rear axle bearing",
    "keywords_en": ["bearing"],
    "part_numbers": [],
    "safety_flag": "normal",
    "is_bypass": False,
    "is_wiring": False,
}


@pytest.mark.parametrize("question_en", ["", "   ", "\n\t"])
def test_a_blank_english_question_is_an_invalid_translation(question_en: str) -> None:
    """A blank English question is a failed translation."""
    with pytest.raises(ValidationError):
        Translation.model_validate({**TRANSLATION_DATA, "question_en": question_en})


@pytest.mark.parametrize("terms", ["", "  "])
@pytest.mark.parametrize("keywords", [[], ["", "  "]])
def test_no_search_text_at_all_is_an_invalid_translation(terms: str, keywords: list[str]) -> None:
    """Search terms and keywords both without text is a failed translation."""
    data = {**TRANSLATION_DATA, "search_terms_en": terms, "keywords_en": keywords}
    with pytest.raises(ValidationError):
        Translation.model_validate(data)


def test_search_terms_or_keywords_alone_are_enough() -> None:
    """One of the two with real text is enough."""
    only_terms = {**TRANSLATION_DATA, "keywords_en": []}
    only_keywords = {**TRANSLATION_DATA, "search_terms_en": ""}
    assert Translation.model_validate(only_terms).search_terms_en == "rear axle bearing"
    assert Translation.model_validate(only_keywords).keywords_en == ["bearing"]


# A person who is burned is an emergency (ADR-30), also without a body part in the text
@pytest.mark.parametrize(
    "text", ["ik heb me verbrand", "hij heeft zich verbrand", "brandwonden aan de arm"]
)
def test_a_burned_person_is_an_emergency(text: str) -> None:
    assert question_level(text) == "emergency"


# ---- A person who is hurt, charge and motor cable, more bypass wordings (fix list 5, 2 to 4) ---


@pytest.mark.parametrize(
    "text",
    [
        # en
        "I burned my hand", "my hand is burnt", "someone is hurt", "I got a shock from the kart",
        "burnt smell", "The mechanic burned himself", "My colleague got burnt",
        "He is burned on the face", "the driver is injured",
        # fr
        "je me suis brule la main", "mon collegue est blesse", "la main est brulee",
        "il s'est brule", "odeur de brule dans le garage", "brulure au bras", "il est electrocute",
        # nl
        "mijn collega is verbrand", "ik heb me lelijk verbrand", "ik ruik iets verbrands",
        "verbrande geur in de pit", "ik heb een schok gekregen", "ik kreeg een schok",
        "hij is gewond", "mij verbrand aan het vuur", "mijn voet is verbrand",
    ],
)  # fmt: skip
def test_a_person_who_is_hurt_is_an_emergency_in_three_languages(text: str) -> None:
    """Burns, shocks and injuries are an emergency in nl, fr and en."""
    assert question_level(text) == "emergency"


@pytest.mark.parametrize(
    "text",
    [
        "burned brake pads", "burnt clutch", "plaquettes de frein brulees",
        "remblokken verbrand", "shock absorber", "Replace the shock absorbers",
        "De schokbreker vervangen", "Une schokdemper",
    ],
)  # fmt: skip
def test_burnt_parts_and_shock_absorbers_stay_normal(text: str) -> None:
    """A burnt part is not a hurt person, and a shock absorber is not a shock."""
    assert question_level(text) == "normal"


@pytest.mark.parametrize(
    "text",
    [
        "the kart won't charge", "le kart ne charge pas",
        "Unplug the charge cable and clean the charge port.", "Debranchez le cable moteur.",
        "Disconnect the phase cables.", "Debranchez le cable de phase.", "laadruimtekabel",
    ],
)  # fmt: skip
def test_charging_and_motor_cables_are_high_voltage(text: str) -> None:
    """Charge and the motor or phase cables are high voltage, as a question or a generated step."""
    assert question_level(text) == "high_voltage"


def test_charge_steps_are_dropped_by_the_check() -> None:
    """The generated steps of the review raise unsafe_words."""
    for text in (
        "Unplug the charge cable and clean the charge port.",
        "Debranchez le cable moteur.",
    ):
        assert step_flags(text, "Unplug it.") == ["unsafe_words"]


@pytest.mark.parametrize(
    "text",
    [
        "noodstop uit", "noodstop uitzetten", "zet de noodstop uit", "remsensor uitzetten",
        "turn off the emergency stop", "switch the emergency stop off", "Turn off the brake sensor",
        "switch off the seat switch", "contourner l'arret d'urgence", "Contournez le capteur",
        "disable the seat switch", "jump the seat switch", "bridge the seat switch",
        "Contournez l'arret d'urgence.", "Disable the seat switch.",
    ],
)  # fmt: skip
def test_more_ways_to_switch_off_a_safety_device_are_a_bypass(text: str) -> None:
    """Question wordings and generated steps."""
    assert is_bypass(text)
    assert step_flags(text, "Nothing.") == ["unsafe_words"]


@pytest.mark.parametrize(
    "text",
    ["Reset de noodstop", "reset the emergency stop", "replace the seat switch",
     "De remsensor vervangen"],
)  # fmt: skip
def test_resetting_or_replacing_a_safety_device_is_not_a_bypass(text: str) -> None:
    assert not is_bypass(text)


def test_an_ignored_word_is_blanked_only_as_a_whole_word() -> None:
    """ "laadruimtekabel" is high voltage through "laad*" only; blanking inside it would hide it."""
    assert question_level("laadruimtekabel") == "high_voltage"
    assert question_level("De laadruimte is vuil") == "normal"


# Other forms of "shock" are an emergency too
@pytest.mark.parametrize(
    "text", ["I was shocked by the kart", "I got shocked", "the kart shocks me"]
)
def test_being_shocked_is_an_emergency(text: str) -> None:
    assert question_level(text) == "emergency"
