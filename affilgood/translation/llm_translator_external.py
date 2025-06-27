import os
import time

import requests
import requests_cache
from requests_cache import CachedSession

from affilgood.translation.llm_translator import LLMTranslator
from affilgood.translation.model import TranslationLLMModel, Translation

EXTERNAL_API_URL = "https://api.together.xyz/v1/chat/completions"
EXTERNAL_API_KEY = ""  # Set API key
EXTERNAL_API_MAX_RETRIES = 3
EXTERNAL_API_RETRY_DELAY = 2  # seconds

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUESTS_CACHE_PATH = os.path.join(CURRENT_DIR, 'translation_http_cache')
CACHE_EXPIRATION = 604800  # 7 days in seconds

TRANSLATION_PROMPT_EXTERNAL = """
Translate the following text to English with EXACT PRECISION. 

CRITICAL RULES:
1. Translate institution names LITERALLY
2. DO NOT substitute any institution with a more famous one
3. Preserve ALL place names and institution names exactly as they appear
4. DO NOT add or remove any information
5. Provide ONLY the direct translation with no explanations

Text to translate:
"""


class LLMTranslatorExternal(LLMTranslator):

    def _perform_translation(self, translation: Translation) -> Translation:
        response = self._call_external_api(translation.original_text)

        if response:
            translation.translated_text = response.strip('"').strip()

        return translation

    def _perform_batch_translation(self, translations: list[Translation], batchsize: int) -> list[Translation]:
        return [self._perform_translation(t) for t in translations]

    def _call_external_api(self, text: str) -> str | None:
        """
        Call an external API for text generation with caching.

        Returns:
            Generated text response from the API
        """
        headers = {
            "Authorization": f"Bearer {self.conf.external_api_key}",
            "Content-Type": "application/json"
        }

        # Get model-specific configuration
        model_config = self._get_model_specific_config(self.conf.translation_model)
        prompt_content = self._mk_prompt(text)

        data = {
            "model": self.conf.translation_model,
            "messages": [
                {"role": "system",
                 "content": "You are a specialized academic translator focusing on institutional affiliations."},
                {"role": "user",
                 "content": prompt_content}
            ],
            **model_config  # Apply all model-specific configs
        }

        # Generate a cache key based on the prompt and model
        # cache_key = f"{self.conf.translation_model}:{self.conf.translation_prompt}"

        cached_session = None

        if self.conf.use_cache:
            cached_session = self._setup_requests_cache()

        # Implement retry logic for API calls
        for attempt in range(EXTERNAL_API_MAX_RETRIES):
            try:
                if self.conf.verbose_logging:
                    print(f"Calling external API (attempt {attempt + 1}/{EXTERNAL_API_MAX_RETRIES})...")

                # Use cached session if available, otherwise use regular requests
                if cached_session and self.conf.use_cache:
                    response = cached_session.post(
                        url=self.conf.external_api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )

                    # Update stats if this was a cache hit
                    if hasattr(response, 'from_cache') and response.from_cache:
                        self.stats.cache_hits += 1
                        if self.conf.verbose_logging:
                            print("Retrieved translation from cache")
                else:
                    response = requests.post(
                        url=self.conf.external_api_url,
                        headers=headers,
                        json=data,
                        timeout=30
                    )

                response.raise_for_status()  # Raise exception for HTTP errors

                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    if self.conf.verbose_logging:
                        print(f"Unexpected API response format: {result}")

            except Exception as e:
                if self.conf.verbose_logging:
                    print(f"API call failed: {str(e)}")

                # Wait before retrying, unless it's the last attempt
                if attempt < EXTERNAL_API_MAX_RETRIES - 1:
                    retry_delay = EXTERNAL_API_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    time.sleep(retry_delay)

        # If all attempts failed
        if self.conf.verbose_logging:
            print("All API call attempts failed")
        return None

    def _setup_requests_cache(self) -> CachedSession:
        """Set up the requests cache for HTTP requests."""
        cached_session = None

        try:
            cache_path = self.conf.cache_path

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # Initialize the cached session
            cached_session = requests_cache.CachedSession(
                cache_name=cache_path,
                backend='sqlite',
                expire_after=self.conf.cache_expire_after
            )

            if self.conf.verbose_logging:
                print(f"HTTP caching initialized at {cache_path}")

        except Exception as e:
            print(f"Warning: Failed to initialize requests_cache: {e}")
            print("Continuing without HTTP caching")

        finally:
            return cached_session

    @staticmethod
    def _get_model_specific_config(translation_model: TranslationLLMModel) -> dict:
        """
        Get model-specific configuration parameters.

        Args:
            translation_model: Name/identifier of the model

        Returns:
            Dictionary with model-specific parameters
        """
        # Base configuration that works for most models
        base_config = {
            'temperature': 0.1,  # Lower temperature for more deterministic output
            'max_tokens': 250,  # Affiliations are short
            'top_p': 0.9,  # Higher top_p for more focused sampling
            'top_k': 40,  # Slightly narrower token selection
            'repetition_penalty': 1.03  # Slight penalty to avoid repetition
        }

        # Model-specific adjustments
        if 'llama' in translation_model.lower():
            return {
                **base_config,
                'stop': ["<|eot_id|>", "<|eom_id|>"]
            }
        elif 'gemma' in translation_model.lower():
            return {
                **base_config,
                'stop': ["<eos>", "<end_of_turn>"]
            }
        elif 'mistral' in translation_model.lower() or 'mixtral' in translation_model.lower():
            return {
                **base_config,
                'stop': ["</s>"]
            }
        elif 'claude' in translation_model.lower():
            return {
                **base_config,
                'stop': ["Human:", "H:"]
            }

        # Default configuration
        return base_config
