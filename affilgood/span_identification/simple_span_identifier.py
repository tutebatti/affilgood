from affilgood.span_identification.model import SplitResult
from affilgood.span_identification.span_identifier_interface import SpanIdentifierInterface


class SimpleSpanIdentifier(SpanIdentifierInterface):
    """
    A simple implementation of span identification that treats each input text
    as a complete span without complex processing.
    
    Can optionally split text by a separator character to create multiple spans.
    """

    def __init__(self, separator: str = ";", **kwargs):
        super().__init__(**kwargs)
        self.separator = separator

    def identify_spans(self) -> None:
        self.results = []

        for raw_text in self.raw_text_list:
            spans = _mk_spans(raw_text=raw_text, separator=self.separator)
            self.results.append(SplitResult(raw_text=raw_text, spans=spans))


def _mk_spans(raw_text: str, separator: str = ";") -> list[str]:
    if separator is None:
        return [raw_text]

    separated_raw_text = raw_text.split(separator)
    spans = [span.strip() for span in separated_raw_text if span.strip()]
    return spans or [raw_text]
