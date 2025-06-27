from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from affilgood.language_prediction.model import LangCode


class TranslationLLMModel(StrEnum):
    LLAMA = "llama"
    GEMMA = "gemma"
    MISTRAL = "mistral"
    CLAUDE = "claude"


def mk_default_request_cache_path() -> Path:
    current_dir = Path(__file__).resolve().parent
    requests_cache_path = current_dir / 'translation_http_cache'
    return requests_cache_path


@dataclass
class TranslationConfig:
    """
    Args:
        skip_english: Whether to skip translation for English text
        translation_model: Name of the Hugging Face model to use
        use_external_api: Whether to use an external API instead of local model
        external_api_url: URL for the external API (if `use_external_api` is True)
        external_api_key: API key for the external API (if `use_external_api` is True)
        verbose_logging: Whether to show detailed loading information
        use_cache: Whether to use HTTP request caching
        cache_expire_after: Cache expiration time in seconds
        cache_path: Path to the cache directory
    """
    translation_model: TranslationLLMModel
    translation_prompt: str = None
    skip_english: bool = True
    use_external_api: bool = False
    external_api_url: str | None = None
    external_api_key: str | None = None
    verbose_logging: bool = False
    use_cache: bool = True
    cache_expire_after: int = 0  # seconds
    cache_path: str | Path = mk_default_request_cache_path()


@dataclass
class Translation:
    predicted_lang: LangCode
    original_text: str
    translated_text: str = None

    @property
    def is_english(self) -> bool:
        return self.predicted_lang == "en"

    @property
    def translation_performed(self) -> bool:
        return self.original_text != self.translated_text
