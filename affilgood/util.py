import re

from affilgood.span_identification.model import TextInput


def preprocess_text_input(text_input: TextInput) -> list[str]:
    text_input = _mk_list_from_str(text_input)
    text_input = [clean_whitespaces(text) for text in text_input]
    return text_input


def mk_title_case(text_list: list[str]) -> list[str]:
    return [text.title() for text in text_list]


def clean_whitespaces(text: str) -> str:
    """Clean extra whitespace from text."""
    return re.sub(r'\s+', ' ', str(text).strip())


def _mk_list_from_str(text_input: TextInput) -> list[str]:
    if isinstance(text_input, str):
        text_input = [text_input]
    return text_input
