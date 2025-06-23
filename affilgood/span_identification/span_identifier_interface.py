from affilgood.span_identification.model import Span, TextInput
from affilgood.util import preprocess_text_input


class SpanIdentifierInterface:
    raw_text_list: list[str]
    spans: list[Span] = []

    def __init__(self, **kwargs) -> None:
        pass

    def set_text_input(self, text_input: TextInput) -> None:
        self.raw_text_list = preprocess_text_input(text_input)

    def identify_spans(self) -> None:
        pass
