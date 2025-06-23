from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from affilgood.span_identification.model import Span
from affilgood.span_identification.span_identifier_interface import SpanIdentifierInterface
from affilgood.util import mk_title_case

# See https://huggingface.co/nicolauduran45/affilgood-span-multilingual-v2
DEFAULT_SPAN_MODEL = 'nicolauduran45/affilgood-span-v2'

DEFAULT_BATCH_SIZE = 64
DEFAULT_THRESHOLD_SCORE = 0.75


class SpanIdentifier(SpanIdentifierInterface):
    def __init__(self,
                 span_model: str = None,  # identifier of Model, e.g. at Hugging Face
                 device: int | str | torch.device | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 threshold_score: float = DEFAULT_THRESHOLD_SCORE,
                 fix_predicted_words: bool = True,
                 title_case: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)

        self._set_device(device)
        self._set_model(span_model)

        self.batch_size: int = batch_size
        self.threshold_score: float = threshold_score
        self.fix_predicted_words: bool = fix_predicted_words
        self.title_case: bool = title_case

    def _set_device(self, device) -> None:
        if device is None and torch.cuda.is_available():
            self.device = 0
        elif device is None:
            self.device = -1
        else:
            self.device = device

    def _set_model(self, span_model: str | Path) -> None:
        if span_model is None:
            self.span_model = DEFAULT_SPAN_MODEL
        else:
            self.span_model = span_model

        self.pipeline = pipeline(
            task="token-classification",
            model=AutoModelForTokenClassification.from_pretrained(self.span_model),
            tokenizer=AutoTokenizer.from_pretrained(self.span_model),
            aggregation_strategy="simple",
            device=self.device
        )

    def identify_spans(self, batch_size: int = None) -> None:
        if batch_size is None:
            batch_size = self.batch_size

        if self.title_case:
            self.raw_text_list = mk_title_case(self.raw_text_list)

        pipeline_outputs = self.pipeline(
            self.raw_text_list,
            batch_size=batch_size
        )

        if len(pipeline_outputs) != len(self.raw_text_list):
            raise RuntimeError("Mismatch between input texts and model pipeline outputs")

        for raw_text, named_entities in zip(self.raw_text_list, pipeline_outputs):

            if self.fix_predicted_words:
                named_entities = _fix_predicted_words(raw_text=raw_text, named_entities=named_entities)

            cleaned_entities = _clean_and_merge_entities(named_entities)

            span_entities = [entity.get("word", "") for entity in cleaned_entities]

            self.spans.append(Span(raw_text=raw_text, named_entities=span_entities))


def _fix_predicted_words(raw_text: str, named_entities: list[dict]):
    for entity in named_entities:
        start, end = entity["start"], entity["end"]
        entity["word"] = raw_text[start:end]
    return named_entities


def _clean_and_merge_entities(entities: list[dict], min_score: float = DEFAULT_THRESHOLD_SCORE) -> list[dict]:
    entities = [entity for entity in entities if entity.get("score", 0) >= min_score]

    merged_entities = []
    i = 0
    while i < len(entities):
        current_entity = entities[i]
        if i + 1 < len(entities):
            next_entity = entities[i + 1]
            if current_entity['end'] == next_entity['start'] and next_entity['word'][0].islower():
                merged_entities.append({
                    "entity_group": current_entity['entity_group'],
                    "score": min(current_entity['score'], next_entity['score']),
                    "word": current_entity['word'] + next_entity['word'],
                    "start": current_entity['start'],
                    "end": next_entity['end']
                })
                i += 2
                continue
        merged_entities.append(current_entity)
        i += 1
    return merged_entities
