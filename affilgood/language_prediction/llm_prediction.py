#!/usr/bin/env python3
from enum import StrEnum

import pycountry
from torch import Tensor
from torch.types import Device

from affilgood.language_prediction.model import LangCode


class LLMType(StrEnum):
    FASTTEXT = "fasttext"
    E5 = "e5"
    LANGDETECT = "langdetect"
    NONE = None


LANGDETECT_SEED = 2


def determine_lang_with_llm(text: str,
                            llm_type: LLMType = None,
                            default_lang: LangCode = "ud"
                            ) -> LangCode:
    """
    Get language prediction using the specified model.
    """
    model = None
    tokenizer = None

    # Ensure the correct model is loaded
    if llm_type is not None and llm_type not in LLMType:
        try:
            print(f"==> Loading model: {llm_type}")
            model, tokenizer, model_probs = _load_llm(llm_type)
        except Exception as e:
            print(f"Error loading LLM type {llm_type}: {e}")
            print(f"Returning default lang {default_lang}")
            return default_lang

    if model is None:
        print(f"Warning: No LLM available for {llm_type}")
        return default_lang

    # Use the appropriate detection function
    if llm_type == "e5":
        return _determine_language_e5(text=text, model=model, tokenizer=tokenizer, default_lang=default_lang)
    elif llm_type == "fasttext":
        return _determine_language_fasttext(text=text, model=model, default_lang=default_lang)
    elif llm_type == "langdetect":
        return _determine_lang_with_langdetect(text=text, model=model, default_lang=default_lang)
    else:
        return default_lang


def _load_llm(model_type: LLMType = "e5") -> tuple:
    """
    Load the specified language detection model.
    """
    model = None
    tokenizer = None

    try:
        if model_type == "e5":
            model, tokenizer = _load_e5()
        elif model_type == "fasttext":
            model, tokenizer = _load_fasttext()
        elif model_type == "langdetect":
            model, tokenizer = _load_langdetect()
        else:
            print(f"No model selected (model_type={model_type}). Using heuristic detection only.")
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")

    # Log the status of the model loading
    if model is None:
        print("No model was loaded successfully.")
    else:
        print(f"Model {model_type} loaded successfully.")

    return model, tokenizer


def _load_e5():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "Mike0307/multilingual-e5-language-detection",
        num_labels=45
    )
    tokenizer = AutoTokenizer.from_pretrained("Mike0307/multilingual-e5-language-detection")
    print("E5 language detection model loaded successfully")

    return model, tokenizer


def _load_fasttext():
    import fasttext
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
    model = fasttext.load_model(model_path)
    print(f"FastText model loaded from {model_path}")
    return model, None


def _load_langdetect():
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = LANGDETECT_SEED  # Set a fixed seed for deterministic results
    model = detect
    print("Package langdetect imported successfully")
    return model, None


def _determine_language_e5(
    text: str,
    model,
    tokenizer,
    default_lang: LangCode = "un"
) -> LangCode:
    """
    Get language prediction using E5 model.

    Args:
        text (str): The text to analyze

    Returns:
        str: Detected language code
    """
    languages = [
        "ar", "eu", "br", "ca", "zh", "zh", "zh", "cv", "cs", "dv",
        "nl", "en", "eo", "et", "fr", "fy", "ka", "de", "el", "cnh",
        "id", "ia", "it", "ja", "kab", "rw", "ky", "lv", "mt", "mn",
        "fa", "pl", "pt", "ro", "rm", "ru", "sah", "sl", "es", "sv",
        "ta", "tt", "tr", "uk", "cy"
    ]

    if model is None:
        print(f"Warning: No model available for e5")
        return default_lang

    probs = _predict_e5(text=text, model=model, tokenizer=tokenizer, device=None)
    topk_prob, topk_labels = _get_topk_e5(probs, languages, k=1)
    lang_code = topk_labels[0] if topk_labels else default_lang

    return lang_code


def _predict_e5(text: str, model, tokenizer, device: Device = None):
    """
    Get language prediction probs using E5 model.

    Args:
        text (str): The text to analyze
        device: PyTorch device to use

    Returns:
        torch.Tensor: Probabilities for each language
    """

    if not hasattr(model, "to"):
        raise ValueError('E5 model not properly loaded (no "to" method)')

    try:
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs

    except Exception as e:
        print(f"Error in _predict_e5: {e}")
        raise


def _get_topk_e5(probabilities, languages, k=3):
    """
    Get top-k language predictions from E5 model.

    Args:
        probabilities (torch.Tensor): Probabilities from predict_e5
        languages (list): List of language codes
        k (int): Number of top predictions to return

    Returns:
        tuple: Lists of top probabilities and language codes
    """
    import torch

    topk_prob, topk_indices = torch.topk(probabilities, k)
    topk_prob = topk_prob.cpu().numpy()[0].tolist()
    topk_indices = topk_indices.cpu().numpy()[0].tolist()
    topk_labels = [languages[index] for index in topk_indices]

    return topk_prob, topk_labels


def _determine_language_fasttext(text: str, model, default_lang: LangCode = "un") -> LangCode:
    """
    Get language prediction using FastText model.

    Args:
        text (str): The text to analyze
        default_lang: "un" for undefined

    Returns:
        str: Detected language code
    """

    if model is None:
        print(f"Warning: No model available for fasttext")
        return default_lang

    # Get three-letter language code by means of FastText
    predictions = model.predict(text)

    # Extract the language label
    label = predictions[0][0]

    # Remove the "__label__" prefix to get the language code
    parts = label.replace("__label__", "").split("_")
    language_code_3chars = parts[0]
    lang_code = _lang_code_3_to_2(language_code_3chars)

    return lang_code


def _lang_code_3_to_2(code_3) -> LangCode:
    """
    Convert ISO 639-3 three-letter language code to ISO 639-1 two-letter code.

    Args:
        code_3 (str): Three-letter language code

    Returns:
        str: Two-letter language code or "und" if not found
    """
    try:
        language = pycountry.languages.get(alpha_3=code_3)
        return language.alpha_2 if hasattr(language, "alpha_2") else LangCode("un")
    except (AttributeError, LookupError):
        return LangCode("un")


def _determine_lang_with_langdetect(
    text: str,
    model,
    default_lang: LangCode = "un"
) -> LangCode:

    if model is None:
        print(f"Warning: No model available for langdetect")
        return default_lang

    try:
        return model(text).split("-")[0]
    except Exception as e:
        print(f"Error detecting language with langdetect: {e}")
        print(f"Returning default language: {default_lang}")
        return default_lang


def get_probs_with_langdetect(text: str) -> dict:
    from langdetect import detect_langs
    model_probs = detect_langs
    try:
        # Returns a list of Language objects with lang and prob attributes
        lang_probabilities = model_probs(text)
        # Convert to a dictionary for easier handling
        result = {lang.lang.split("-")[0]: lang.prob for lang in lang_probabilities}
        return result
    except Exception as e:
        print(f"Error detecting language with langdetect: {e}")
        print(f"Returning empty probs")
        return {}
