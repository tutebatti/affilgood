import io
import logging
import sys
import warnings
from typing import Any

from transformers import pipeline, logging as transformers_logging, Pipeline

from affilgood.language_prediction.heuristic_prediction import predict_lang_heuristically
from affilgood.translation.llm_translator import LLMTranslator
from affilgood.translation.model import Translation

DISABLE_HF_OUTPUT = False
HF_TOKEN = ""
MODEL_REQUIRES_AUTHENTICATION = False

MAX_NEW_TOKENS = 500  # Adjust based on expected output length
DEFAULT_BATCH_SIZE = 8  # Default batch size for GPU processing

REASONABLE_LENGTH = 3

TRANSLATION_PROMPT_LOCAL = """
You are a specialized academic translator focusing on institutional affiliations. 

TRANSLATION PROCESS:
1. FIRST, identify any institution names, cities, or locations in the original text
2. Research the standard English spelling of these proper nouns
3. THEN translate the whole text to English while preserving these identified entities

TRANSLATION RULES:
- Keep all university names, research centers, and geographical locations in their standard English form
- Translate academic titles and departments accurately
- Maintain the original text structure

Text to translate:
"""


class LLMTranslatorLocal(LLMTranslator):
    ppln: Pipeline
    pad_token_id: Any

    def __post_init__(self) -> None:
        self._load_pipeline()
        self.pad_token_id = self.ppln.tokenizer.eos_token_id

    def _perform_translation(self, translation: Translation) -> Translation:
        prompt: str = self._mk_prompt(text=translation.original_text)

        # Add safeguards against repeated prompt patterns
        try:
            outputs = self.ppln(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.pad_token_id,
                num_return_sequences=1  # Get only one output sequence
            )

            # Extract translated text
            response = outputs[0]['generated_text'].replace(prompt, '').strip()
            translated_text = self._clean_response(response)

            # Verify the output is reasonable
            if translated_text and len(translated_text) >= REASONABLE_LENGTH:
                translation.translated_text = translated_text

            else:
                if self.conf.verbose_logging:
                    print(f"Warning: Translation produced empty or very short result. Using original text.")

        except Exception as e:
            if self.conf.verbose_logging:
                print(f"Translation error: {str(e)}")

        return translation

    def _perform_batch_translation(self, translations: list[Translation], batch_size: int):

        non_english_translations = []

        # Check language for all texts first
        for t in translations:
            t.predicted_lang = predict_lang_heuristically(t.original_text)
            if not t.is_english:
                non_english_translations.append(t)

        batch = []

        for i, nte in non_english_translations:

            batch.append(nte)

            if i % batch_size == 0:

                self._translate_batch(batch=batch)
                batch.clear()

        self._translate_batch(batch=batch)

    def _translate_batch(self, batch: list[Translation]):

        try:
            prompts = [self._mk_prompt(text=translation.original_text) for translation in batch]

            # Process the batch in a single ppln call
            outputs = self.ppln(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.pad_token_id,
                batch_size=len(prompts)
            )

            for i, (output, translation) in enumerate(zip(outputs, batch)):

                response = output['generated_text'].replace(prompts[i], '').strip()
                translated_text = self._clean_response(response)

                if translated_text and len(translated_text) > REASONABLE_LENGTH:
                    translation.translated_text = translated_text

        except Exception as e:
            if self.conf.verbose_logging:
                print(f"Batch translation error: {str(e)}")
            # Fall back to individual processing for this batch
            for translation in batch:
                try:
                    self._perform_translation(translation=translation)
                except Exception as e:
                    print(f"Exception occured: {e}")
                    # If individual translation fails, use original text

    def _load_pipeline(self) -> None:
        """Load the LLM model with appropriate logging controls."""
        if self.conf.verbose_logging:
            print(f"Loading local LLM translation model: {self.conf.translation_model}")

        if MODEL_REQUIRES_AUTHENTICATION:
            try:
                from huggingface_hub import login
                login(HF_TOKEN)
            except Exception as e:
                print(str(e))

        if DISABLE_HF_OUTPUT:
            # Store original logging levels
            original_tf_verbosity = transformers_logging.get_verbosity()
            original_logging_level = logging.getLogger().level
            try:
                # Disable all transformers logging
                transformers_logging.set_verbosity_error()
                # Suppress other logging
                logging.getLogger().setLevel(logging.ERROR)
                # Disable warnings
                warnings.filterwarnings("ignore")
                # Redirect stdout/stderr during model loading
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    ppln = pipeline('text-generation', model=self.conf.translation_model, device_map="auto")
                finally:
                    # Restore stdout/stderr
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            finally:
                # Restore original logging levels
                transformers_logging.set_verbosity(original_tf_verbosity)
                logging.getLogger().setLevel(original_logging_level)
        else:
            ppln = pipeline('text-generation', model=self.conf.translation_model, device_map="auto")

        if self.conf.verbose_logging:
            print(f"LLM translation model loaded successfully")

        self.ppln = ppln

    @staticmethod
    def _clean_response(response: str) -> str:
        """Clean up the LLM response to extract only the translation."""

        # Handle repeated user/assistant patterns that might appear in the output
        if "<|user|>" in response or "<|assistant|>" in response:
            # Extract only the first meaningful response before any repeated patterns
            parts = response.split("<|user|>")
            response = parts[0].strip()

        # Remove any explanations or additional text that might follow the translation
        if "Input:" in response:
            response = response.split("Input:")[0]

        # Remove any Markdown formatting, etc.
        response = response.replace("*", "").replace("#", "").replace("`", "")

        # Remove any prefix like "Output:" or "Translation:"
        prefixes = ["Output:", "Translation:", "Translated text:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        return response.strip('"').strip()
