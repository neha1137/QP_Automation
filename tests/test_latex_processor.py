"""
tests/test_latex_processor.py — unit tests for the isolated LaTeX
detection/conversion layer (latex_processor.py). Covers the exact
scenarios called out in the feature spec: plain text is left alone,
common notation converts correctly, math embedded in a sentence is
converted without swallowing the whole sentence, ambiguous text is
preserved verbatim, and processing is idempotent (never double-converts).
"""

from __future__ import annotations

from latex_processor import process_text, process_question_fields


# -- 1. Plain text is never touched -----------------------------------------

def test_plain_sentence_unchanged():
    assert process_text("This is a normal sentence.") == "This is a normal sentence."


def test_ambiguous_short_text_unchanged():
    for s in ["AB", "x", "No. 2", "23", "Q49", "12/25/2024"]:
        assert process_text(s) == s


def test_value_of_x_is_5_unchanged():
    """x alone, with no math signal, is not a formula — spec's explicit
    example of what must NOT be converted."""
    assert process_text("The value of x is 5.") == "The value of x is 5."


def test_ordinary_slash_in_a_sentence_unchanged():
    assert process_text("Choose either option A/B before submitting.") == (
        "Choose either option A/B before submitting."
    )


def test_and_or_slash_word_unchanged():
    assert process_text("Bring a pen and/or pencil.") == "Bring a pen and/or pencil."


# -- 2/3. Superscripts -------------------------------------------------------

def test_superscript_square():
    assert process_text("x²") == "$x^2$"


def test_superscript_cube():
    assert process_text("x³") == "$x^3$"


def test_superscript_multidigit_exponent():
    assert process_text("x¹²") == "$x^{12}$"


# -- 4. Subscripts ------------------------------------------------------------

def test_subscript_single_digit():
    assert process_text("a₁") == "$a_1$"


# -- 5. Square roots ----------------------------------------------------------

def test_sqrt_bare_variable():
    assert process_text("√x") == r"$\sqrt{x}$"


def test_sqrt_bare_number():
    assert process_text("√16") == r"$\sqrt{16}$"


def test_sqrt_grouped_expression():
    assert process_text("√(a² + b²)") == r"$\sqrt{a^2 + b^2}$"


# -- 6. Math inside a normal sentence — only the math portion converts -------

def test_math_inside_sentence_only_math_converted():
    assert process_text("The value of x² + y² is 25.") == (
        "The value of $x^2 + y^2$ is 25."
    )


def test_multi_clause_sentence_two_independent_spans():
    result = process_text("If x² + y² = 25 and x = 3, find y.")
    assert result == "If $x^2 + y^2 = 25$ and $x = 3$, find y."


# -- 7. Options A-D work independently ---------------------------------------

def test_option_plain_number_unchanged():
    assert process_text("2") == "2"


def test_option_superscript():
    assert process_text("x²") == "$x^2$"


def test_option_sqrt():
    assert process_text("√16") == r"$\sqrt{16}$"


def test_option_whole_field_fraction():
    assert process_text("16/2") == r"$\frac{16}{2}$"


def test_option_letter_fraction():
    assert process_text("a/b") == r"$\frac{a}{b}$"


def test_process_question_fields_covers_all_option_letters():
    q = {
        "stem": "If x² = 16, find x.",
        "options": {"A": "2", "B": "4", "C": "√16", "D": "16/2"},
        "explanation": "",
        "is_image": False,
    }
    process_question_fields(q)
    assert q["stem"] == "If $x^2 = 16$, find x."
    assert q["options"]["A"] == "2"
    assert q["options"]["B"] == "4"
    assert q["options"]["C"] == r"$\sqrt{16}$"
    assert q["options"]["D"] == r"$\frac{16}{2}$"


# -- 8. Explanations work -----------------------------------------------------

def test_explanation_field_converted():
    q = {
        "stem": "s", "options": {"A": "", "B": "", "C": "", "D": ""},
        "explanation": "Using a² + b² = c², we get c = 5.", "is_image": False,
    }
    process_question_fields(q)
    # "c = 5" is itself a bare letter=number equation — also math, same as
    # spec's own "x = 3" example — so both spans convert independently.
    assert q["explanation"] == "Using $a^2 + b^2 = c^2$, we get $c = 5$."


# -- 9. Ambiguous text not aggressively converted ----------------------------

def test_no_conversion_without_a_strong_math_signal():
    for s in [
        "Rearrange the letters to form a word.",
        "Section B contains 20 questions.",
        "Directions: read the passage carefully.",
    ]:
        assert process_text(s) == s


# -- 10. Existing LaTeX is never double-converted ----------------------------

def test_existing_latex_left_untouched():
    assert process_text("$x^2$") == "$x^2$"


def test_existing_latex_inside_sentence_left_untouched():
    text = "The formula is $x^2 + y^2 = z^2$ as shown."
    assert process_text(text) == text


def test_reprocessing_output_is_idempotent():
    once = process_text("x² + y² = z²")
    twice = process_text(once)
    assert once == twice == "$x^2 + y^2 = z^2$"


def test_partial_latex_plus_new_math_only_converts_the_new_part():
    text = "$x^2$ and also y³"
    result = process_text(text)
    assert result == "$x^2$ and also $y^3$"


# -- Inequalities / multiplication / division symbols ------------------------

def test_inequality_and_symbols():
    assert process_text("x ≤ 5") == r"$x \le 5$"
    assert process_text("x ≥ 5") == r"$x \ge 5$"
    assert process_text("x ≠ 5") == r"$x \neq 5$"
    assert process_text("6 × 7") == r"$6 \times 7$"
    assert process_text("6 ÷ 7") == r"$6 \div 7$"


# -- None / empty safety ------------------------------------------------------

def test_none_and_empty_pass_through():
    assert process_text("") == ""
    assert process_text(None) is None
