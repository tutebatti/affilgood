from dataclasses import dataclass

TextInput = str | list[str]


@dataclass
class SplitResult:
    raw_text: str
    spans: list[str]
