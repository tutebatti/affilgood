import time
from dataclasses import dataclass, field

from affilgood.language_prediction.heuristic_prediction import predict_lang_heuristically
from affilgood.language_prediction.model import LangCode
from affilgood.translation.model import Translation, TranslationConfig
from affilgood.translation.translation_stats import TranslationStats

# DEFAULT_MODEL = "TheBloke/neural-chat-7B-v3-2-GPTQ"
DEFAULT_MODEL_LOCAL = "google/gemma-3-27b-it"
DEFAULT_MODEL_EXTERNAL = "google/gemma-2-27b-it"

USE_EXTERNAL_API = False


@dataclass
class LLMTranslator:
    conf: TranslationConfig
    input_texts: list[str] = field(default_factory=list)
    results: list[Translation] = field(default_factory=list)
    stats: TranslationStats = field(default_factory=TranslationStats)

    def process(self, text: str) -> Translation:
        """Translates affiliation strings from any language to English using an LLM."""

        start_time = time.time()

        preprocessed_text = self._preprocess_text(text=text)
        predicted_lang = predict_lang_heuristically(text=text)

        translation = Translation(
            original_text=preprocessed_text,
            predicted_lang=predicted_lang
        )

        if self.conf.skip_english and predicted_lang == "en":
            return translation

        translation = self._perform_translation(translation)

        # Update stats
        translation_time = time.time() - start_time
        processing_time = translation_time
        self.stats.update(1, 1, 0, translation_time, processing_time)

        if self.conf.verbose_logging and translation.translation_performed:
            print(f"Original: {translation.original_text}")
            print(f"Translated: {translation.translated_text}")

        return translation

    def process_batch(self, texts: list[str], batch_size: int) -> list[Translation]:
        """
        Translate a batch of texts efficiently.

        Args:
            texts: List of texts to translate
            batch_size: Number of texts to process in parallel (depends on GPU memory)

        Returns:
            List of translated texts
        """
        start_time = time.time()

        if not texts:
            return []

        # Filter out empty or None texts
        preprocessed_texts = []
        for t in texts:
            text = self._preprocess_text(t)
            if text is not None:
                preprocessed_texts.append(text)

        if not preprocessed_texts:
            return [Translation(original_text="", predicted_lang=LangCode("un"))] * len(texts)

        translations = [Translation(original_text=t, predicted_lang=LangCode("un")) for t in preprocessed_texts]

        results = self._perform_batch_translation(translations=translations, batch_size=batch_size)

        # Update stats
        translation_time = time.time() - start_time
        processing_time = translation_time

        self.stats.update(
            processed=len(preprocessed_texts),
            performed=sum(1 for t in translations if t.translation_performed),
            cache_hits=0,
            translation_time=translation_time,
            processing_time=processing_time
        )

        # Map results back to original text list (including empty ones)
        final_results = []
        valid_idx = 0
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                final_results.append(results[valid_idx])
                valid_idx += 1
            else:
                final_results.append(text)

        return final_results

    @staticmethod
    def _preprocess_text(text: str) -> str | None:
        text = text.strip()

        if not text or not isinstance(text, str):
            return None

        return text

    def _mk_prompt(self, text: str) -> str:
        return f'{self.conf.translation_prompt}\n"{text}"\n'

    def _perform_translation(self, unprocessed_translation: Translation) -> Translation:
        pass

    def _perform_batch_translation(self, translations: list[Translation], batch_size: int) -> list[Translation]:
        pass
