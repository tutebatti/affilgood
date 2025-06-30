from affilgood.language_prediction.heuristic_prediction import predict_lang_heuristically
from affilgood.language_prediction.llm_prediction import LLMType, predict_lang_with_llm, get_probs_with_langdetect
from affilgood.language_prediction.model import LangCode


def predict_lang_with_langdetect_and_heuristic(text: str, default_lang: LangCode = "en") -> LangCode:
    """
    Advanced combination of heuristic detection and langdetect probabilities.
    """

    # Get heuristic language prediction
    heuristically_determined_lang = predict_lang_heuristically(text, default_lang=default_lang)

    # Get language probabilities from langdetect
    lang_probs = get_probs_with_langdetect(text=text)

    # Get top langdetect prediction
    langdetect_top = max(lang_probs.items(), key=lambda x: x[1])[0] if lang_probs else default_lang

    if langdetect_top == "ca":
        return LangCode("ca")

    if langdetect_top == "nl" and heuristically_determined_lang != "en":
        return LangCode("nl")

    if heuristically_determined_lang in ["en", "es"]:
        return heuristically_determined_lang

    if langdetect_top not in ["ja", "ko", "zh"]:
        return langdetect_top

    if langdetect_top in ["ja", "ko", "zh"]:
        if langdetect_top == "ja" or heuristically_determined_lang == "ja":
            return LangCode("ja")
        if langdetect_top == "zh":
            return LangCode("zh")
        # Check second best langdetect option
        second_best = sorted(list(lang_probs.items()), key=lambda x: x[1], reverse=True)
        if len(second_best) > 1 and second_best[1][0] == "zh":
            return LangCode("zh")  # Chinese was second choice
        # Default option is "ja"
        return LangCode("ja")

    if langdetect_top == heuristically_determined_lang:
        return langdetect_top

    else:
        return default_lang


def predict_lang_with_e5_and_heuristic(text: str, llm_type="e5", default_lang: LangCode = "un") -> LangCode:
    """
    Two-step language detection:
    1. Try enhanced heuristic detection first
    2. If it fails or returns "und", use model-based detection

    Args:
        text (str): The text to analyze
        llm_type (str): Type of model to use for fallback ("fasttext", "e5")
        default_lang: "un" for undefined

    Returns:
        str: Two-letter language code or "un" if undetermined
    """
    if not isinstance(text, str) or text.strip() == "":
        return default_lang

    # First try the enhanced heuristic detection with "un" as default
    lang_heur = predict_lang_heuristically(text=text, default_lang=default_lang)

    # Use model-based detection only when heuristic is uncertain
    if lang_heur == default_lang:
        try:
            # Use language model detection as fallback
            lang_code = predict_lang_with_llm(text, LLMType[llm_type])
            return lang_code
        except Exception as e:
            print(f"Model detection failed: {e}")
            return lang_heur
    else:
        return lang_heur
