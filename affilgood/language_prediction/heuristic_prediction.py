#!/usr/bin/env python3

import unicodedata
from typing import Callable

import affilgood.language_prediction.latin_heuristic
from affilgood.language_prediction.east_asian_heuristic import evaluate_east_asian
from affilgood.language_prediction.latin_heuristic import evaluate_latin_script
from affilgood.language_prediction.model import LangCode
from affilgood.language_prediction.non_latin_heuristic import evaluate_non_latin_scripts, UNICODE_SCRIPT_RANGES


def predict_lang_heuristically(text: str, default_lang: LangCode = 'un') -> LangCode:
    """
    Enhanced language detection using multiple features
    """
    if not isinstance(text, str) or text.strip() == '':
        return default_lang

    # Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # Handle empty text or text with only ASCII digits/punctuation
    if not text or all(ord(c) < 128 and (c.isdigit() or not c.isalnum()) for c in text):
        return default_lang

    east_asian_evaluation = evaluate_east_asian(text)
    if east_asian_evaluation is not None:
        return east_asian_evaluation

    non_latin_evaluation = evaluate_non_latin_scripts(text)
    if non_latin_evaluation is not None:
        return non_latin_evaluation

    if not _is_latin_script(text):
        return default_lang

    latin_evaluation = evaluate_latin_script(text)
    if latin_evaluation is not None:
        return latin_evaluation

    # Fall back to default value if there are no accented characters
    if not any(char in 'áéíóúàèìòùâêîôûäëïöü' for char in text.lower()):
        return default_lang

    # Fall back to default value if nothing else matches
    return default_lang


def register_language(lang_code: str,
                      specific_chars: set[str] = None,
                      incompatible: set[str] = None,
                      bigrams: list[str] = None,
                      trigrams: list[str] = None,
                      function_words: dict[str: int] = None,
                      academic_kws: dict[str: int] = None,
                      script_ranges: list[tuple[str, str]] = None,
                      pattern_function: Callable = None):
    """
    Register a new language in the detection system.

    Args:
        lang_code (str): ISO 639-1 two-letter language code
        specific_chars (set): Set of language-specific characters
        incompatible (set): Set of incompatible characters
        bigrams (list): List of common bigrams
        trigrams (list): List of common trigrams
        function_words (dict): Dictionary mapping common words to weights
        academic_kws (dict): Dictionary mapping academic words to weights
        script_ranges (list): List of Unicode code point ranges (tuples)
        pattern_function (callable): Function that takes text and returns bool if pattern matches
    """

    # Register language-specific characters
    if specific_chars:
        affilgood.preprocessing.non_latin_heuristic.LANGUAGE_SPECIFIC_CHARS[lang_code] = specific_chars

    # Register incompatible characters
    if incompatible:
        affilgood.preprocessing.non_latin_heuristic.INCOMPATIBLE_CHARS[lang_code] = incompatible

    # Register bigrams
    if bigrams:
        affilgood.preprocessing.n_gram_heuristic.LANGUAGE_BIGRAMS[lang_code] = bigrams

    # Register trigrams
    if trigrams:
        affilgood.preprocessing.n_gram_heuristic.LANGUAGE_TRIGRAMS[lang_code] = trigrams

    # Register function words
    if function_words:
        affilgood.preprocessing.latin_heuristic.COMMON_FUNCTION_WORDS[lang_code] = function_words

    # Register academic words
    if academic_kws:
        affilgood.preprocessing.latin_heuristic.ACADEMIC_KEYWORDS[lang_code] = academic_kws

    # Register script ranges
    if script_ranges:
        UNICODE_SCRIPT_RANGES[lang_code] = script_ranges

    # Register pattern function
    if pattern_function:
        affilgood.preprocessing.latin_heuristic.LANGUAGE_PATTERNS[lang_code] = pattern_function

    print(f"Language '{lang_code}' registered successfully")


def _is_latin_script(text: str) -> bool:
    """
    Check if text contains only Latin characters (including accented).

    Args:
        text (str): The text to analyze

    Returns:
        bool: True if text is Latin script, False otherwise
    """
    # Check if all characters are within Latin ranges or not alphabetic
    return all(
        (not c.isalpha()) or  # Skip non-alphabetic characters
        (ord(c) < 0x0530) or  # Basic Latin, Latin-1 Supplement, Latin Extended A/B
        (0x1E00 <= ord(c) <= 0x1EFF)  # Latin Extended Additional
        for c in text
    )
