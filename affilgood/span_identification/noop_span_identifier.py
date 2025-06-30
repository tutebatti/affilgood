from affilgood.span_identification.model import SplitResult
from affilgood.span_identification.span_identifier_interface import SpanIdentifierInterface


class NoopSpanIdentifier(SpanIdentifierInterface):
    """
    A span identifier that doesn't modify the input text.
    Each input text is treated as a single span without any splitting or modification.
    Useful for pre-segmented data where each input is already a single span.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def identify_spans(self) -> None:
        self.results = [SplitResult(raw_text=raw_text, spans=[raw_text]) for raw_text in self.raw_text_list]
