from abc import ABC

from affilgood.span_identification.model import SplitResult, TextInput


class RawAffiliationStringSplitter(ABC):

    def __init__(self, **kwargs) -> None:
        pass

    def identify_spans(self, raw_text_list: list[TextInput]) -> list[SplitResult]:
        pass


class NoopRawAffiliationStringSplitter(RawAffiliationStringSplitter):
    """
    A span identifier that doesn't modify the input text.
    Each input text is treated as a single span without any splitting or modification.
    Useful for pre-segmented data where each input is already a single span.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def identify_spans(self, raw_text_list: list[TextInput]) -> list[SplitResult]:
        return [SplitResult(raw_text=raw_text, spans=[raw_text]) for raw_text in raw_text_list]


class SimpleRawAffiliationStringSplitter(RawAffiliationStringSplitter):
    """
    A simple implementation of span identification that treats each input text
    as a complete span without complex processing.

    Can optionally split text by a separator character to create multiple spans.
    """

    def __init__(self, separator: str = ";", **kwargs):
        super().__init__(**kwargs)
        self.separator = separator

    def identify_spans(self, raw_text_list = list[TextInput]) -> list[SplitResult]:
        return [
            SplitResult(
                raw_text=raw_text,
                spans=[t.strip() for t in raw_text.split(sep=self.separator)]
            ) for raw_text in raw_text_list
        ]
