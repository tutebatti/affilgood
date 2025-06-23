from dataclasses import dataclass

TextInput = str | list[str]


@dataclass
class Span:
    raw_text: str
    named_entities: list[str]
