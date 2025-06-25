import re
import unicodedata

from affilgood.language_prediction.model import LangCode

# Common bigrams (letter pairs) for various languages
LANGUAGE_BIGRAMS = {
    'en': ['th', 'he', 'in', 'er', 'an', 'ed', 'on', 're', 'at', 'es'],
    'de': ['ch', 'ei', 'ie', 'sc', 'en', 'er', 'in', 'nd', 'te', 'st'],
    'fr': ['ai', 'es', 'le', 'ou', 'qu', 'en', 'on', 'nt', 're', 'de'],
    'es': ['de', 'en', 'qu', 'er', 'es', 'ar', 'la', 'os', 'el', 'ue'],
    'it': ['ch', 'di', 'la', 'co', 'to', 'ri', 'ti', 'er', 'in', 'no'],
    'pt': ['de', 'ar', 'os', 'qu', 'er', 'es', 'do', 'da', 'ão', 'en'],
    'nl': ['en', 'de', 'er', 'ee', 'ij', 'aa', 'an', 'ge', 'ie', 'te'],
    'sv': ['en', 'er', 'et', 'ar', 'de', 'an', 'tt', 'om', 'på', 'fö'],
    'no': ['en', 'er', 'et', 'de', 'om', 'ar', 'st', 'og', 'på', 'il'],
    'da': ['en', 'er', 'et', 'de', 'ar', 'st', 'fo', 'og', 'at', 'af'],
    'fi': ['in', 'en', 'si', 'is', 'an', 'ss', 'aa', 'll', 'ui', 'tä'],
    'pl': ['ie', 'ni', 'cz', 'rz', 'pr', 'zy', 'po', 'na', 'sz', 'ow'],
    'cs': ['vá', 'st', 'ní', 'ro', 'po', 'je', 'né', 'př', 'ho', 'sk'],
    'hu': ['sz', 'el', 'gy', 'en', 'eg', 'na', 'es', 'et', 'ek', 'le'],
    'tr': ['ar', 'in', 'er', 'en', 'an', 'le', 'bi', 'ir', 'ün', 'ka'],
    'ro': ['ul', 'in', 'er', 'ar', 'nt', 're', 'at', 'de', 'la', 'și'],
    'id': ['ng', 'an', 'en', 'me', 'er', 'ka', 'di', 'in', 'be', 'se'],  # Indonesian
    'ja': ['のア', 'のサ', 'した', 'ます', 'てい', 'たち', 'です', 'する', 'いる', 'れる'],
}

# Language-specific trigrams
LANGUAGE_TRIGRAMS = {
    'en': ['the', 'and', 'ing', 'ion', 'ent', 'ati', 'for', 'her', 'ter', 'hat'],
    'es': ['ión', 'ent', 'que', 'ade', 'aci', 'est', 'con', 'ien', 'tra', 'por'],
    'nl': ['een', 'van', 'sch', 'ing', 'ver', 'oor', 'aan', 'den', 'nde', 'eer'],
    'fr': ['ent', 'ion', 'que', 'les', 'ati', 'our', 'ait', 'ans', 'ant', 'lle'],
    'de': ['ein', 'sch', 'die', 'und', 'der', 'che', 'ung', 'eit', 'ich', 'gen'],
    'it': ['ent', 'one', 'che', 'del', 'ato', 'con', 'are', 'ell', 'lla', 'ion'],
    'pt': ['ent', 'çao', 'ção', 'nto', 'ade', 'com', 'ara', 'est', 'que', 'ito'],
    'id': ['ang', 'men', 'eng', 'kan', 'ber', 'ara', 'nga', 'yan', 'ter', 'ata'],  # Indonesian
}


def score_bigrams(text: str) -> dict[LangCode: float]:
    """
    Score text based on language-specific bigrams.

    Args:
        text (str): The text to analyze

    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    text_lower = text.lower()
    if not text_lower:
        return {}

    lang_scores = {lang: 0.0 for lang in LANGUAGE_BIGRAMS}

    # Count bigram occurrences for each language
    for lang, bigrams in LANGUAGE_BIGRAMS.items():
        for bigram in bigrams:
            lang_scores[lang] += text_lower.count(bigram)

    # Normalize by text length to avoid bias from longer texts
    text_len = max(1, len(text))
    for lang in lang_scores:
        lang_scores[lang] = (lang_scores[lang] / text_len) * 100

    return {k: v for k, v in lang_scores.items() if v > 0}


def score_ngrams(text: str, n: int, n_gram_dict: dict[LangCode: list[str]]) -> dict[LangCode: float]:
    """
    Score text based on language-specific n-grams.

    Args:
        text (str): The text to analyze
        n (int): Size of n-gram
        n_gram_dict (dict): Mapping of language code to a list of frequent n-grams in that language

    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    text_n_grams = _extract_ngrams(text=text, n=n)
    if not text_n_grams:
        return {}

    lang_scores = {lang: 0.0 for lang in n_gram_dict}

    # Count trigram occurrences for each language
    for lang, n_grams in n_gram_dict.items():
        for trigram in n_grams:
            lang_scores[lang] += text_n_grams.count(trigram)

    # Normalize by the number of n_grams to avoid bias from longer texts
    text_len = max(1, len(text_n_grams))
    for lang in lang_scores:
        lang_scores[lang] = (lang_scores[lang] / text_len) * 100

    return {k: v for k, v in lang_scores.items() if v > 0}


def _extract_ngrams(text: str, n: int = 3) -> list[str]:
    """
    Extract n-grams from text.

    Args:
        text (str): The text to analyze
        n (int): Size of n-grams (2 for bigrams, 3 for trigrams, etc.)

    Returns:
        list: List of n-grams
    """
    # Normalize and clean the text
    text = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r'\w+', text)

    # Extract character n-grams from each word
    char_n_grams = []
    for word in words:
        if len(word) >= n:
            for i in range(len(word) - n + 1):
                char_n_grams.append(word[i:i+n])

    return char_n_grams
