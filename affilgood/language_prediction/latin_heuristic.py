import re
from collections import Counter

from affilgood.language_prediction.model import LangCode
from affilgood.language_prediction.n_gram_heuristic import score_bigrams, score_ngrams, LANGUAGE_TRIGRAMS
from affilgood.language_prediction.non_latin_heuristic import LANGUAGE_SPECIFIC_CHARS, INCOMPATIBLE_CHARS


def evaluate_latin_script(text: str) -> LangCode | None:
    # For Latin scripts, combine multiple features

    CHAR_WEIGHT = 2
    BIGRAM_WEIGHT = 1
    TRIGRAM_WEIGHT = 1.5
    FUNCTION_WORD_WEIGHT = 10
    ACADEMIC_WORD_WEIGHT = 2.5
    LANGUAGE_PATTERN_WEIGHT = 50

    # Initialize combined scores
    combined_scores = {}

    # 1. Language-specific characters
    char_scores = _score_language_chars(text)
    for lang, score in char_scores.items():
        combined_scores[lang] = combined_scores.get(lang, 0) + score * CHAR_WEIGHT

    # 2. Bigrams
    bigram_scores = score_bigrams(text)
    for lang, score in bigram_scores.items():
        combined_scores[lang] = combined_scores.get(lang, 0) + score * BIGRAM_WEIGHT

    # 3. Trigrams
    trigram_scores = score_ngrams(text=text, n=3, n_gram_dict=LANGUAGE_TRIGRAMS)
    for lang, score in trigram_scores.items():
        combined_scores[lang] = combined_scores.get(lang, 0) + score * TRIGRAM_WEIGHT

    # 4. Common function words
    function_word_scores = _score_common_words(text, COMMON_FUNCTION_WORDS)
    for lang, score in function_word_scores.items():
        combined_scores[lang] = combined_scores.get(lang, 0) + score * FUNCTION_WORD_WEIGHT

    # 5. Academic keywords (especially important for affiliation strings)
    academic_word_scores = _score_academic_keywords(text)
    for lang, score in academic_word_scores.items():
        combined_scores[lang] = combined_scores.get(lang, 0) + score * ACADEMIC_WORD_WEIGHT

    # Apply language patterns for better precision on specific cases
    for lang, pattern_fn in LANGUAGE_PATTERNS.items():
        if pattern_fn(text):
            combined_scores[lang] = combined_scores.get(lang, 0) + LANGUAGE_PATTERN_WEIGHT

    combined_scores = _refine_combined_scores(text, combined_scores)

    # Select language with highest combined score
    if combined_scores:
        return max(combined_scores, key=combined_scores.get)

    return None


COMMON_FUNCTION_WORDS = {
    'en': {
        'the': 10, 'of': 9, 'and': 9, 'to': 8, 'in': 8, 'for': 7, 'at': 7, 'with': 6,
        'by': 6, 'from': 5, 'as': 5, 'on': 5, 'this': 4, 'that': 4
    },
    'es': {
        'el': 10, 'la': 10, 'los': 9, 'las': 9, 'de': 9, 'en': 8, 'y': 8, 'a': 8,
        'que': 7, 'por': 7, 'con': 7, 'para': 6, 'un': 6, 'una': 6, 'se': 5, 'del': 5
    },
    'id': {
        'yang': 10, 'dan': 9, 'di': 9, 'ke': 8, 'pada': 8, 'untuk': 7, 'dengan': 7,
        'dari': 7, 'ini': 6, 'itu': 6, 'oleh': 5, 'atau': 5, 'tidak': 4, 'dalam': 4
    },
    'ja': {
        'の': 10, 'に': 9, 'は': 9, 'を': 8, 'が': 8, 'と': 7, 'で': 7, 'から': 6,
        'より': 6, 'まで': 5, 'など': 5, 'による': 4, 'において': 4
    },
    'nl': {
        'de': 10, 'het': 10, 'een': 9, 'en': 9, 'van': 9, 'in': 8, 'op': 8, 'voor': 7,
        'met': 7, 'door': 6, 'aan': 6, 'bij': 5, 'als': 5, 'uit': 4, 'over': 4
    },
    'fr': {
        'le': 10, 'la': 10, 'les': 9, 'de': 9, 'à': 8, 'des': 8, 'et': 8, 'en': 7,
        'un': 7, 'une': 6, 'du': 6, 'par': 5, 'pour': 5, 'avec': 4, 'sur': 4
    },
    'de': {
        'der': 10, 'die': 10, 'das': 9, 'und': 9, 'in': 8, 'von': 8, 'mit': 7,
        'für': 7, 'auf': 6, 'zu': 6, 'aus': 5, 'bei': 5, 'nach': 4, 'über': 4
    }
}

# Common function words (articles, prepositions, conjunctions) for various languages
LANGUAGE_PATTERNS = {
    'de': lambda t: ('ß' in t or 'ch' in t.lower()) and any(c in LANGUAGE_SPECIFIC_CHARS['de'] for c in t),
    'fr': lambda t: any(c in 'çÇ' for c in t) or ('eau' in t.lower() or 'aux' in t.lower()),
    'es': lambda t: any(c in 'ñÑ¿¡' for c in t) or ('ll' in t.lower() or 'rr' in t.lower()),
    'it': lambda t: ('cch' in t.lower() or 'zz' in t.lower()) and any(c in LANGUAGE_SPECIFIC_CHARS['it'] for c in t),
    'pt': lambda t: any(c in 'ãõÃÕ' for c in t) or ('ção' in t.lower()),
    'sv': lambda t: any(c in 'åÅ' for c in t),
    'no': lambda t: ('og' in t.lower() or 'på' in t.lower()) and any(c in LANGUAGE_SPECIFIC_CHARS['no'] for c in t),
    'da': lambda t: ('og' in t.lower() or 'af' in t.lower()) and any(c in LANGUAGE_SPECIFIC_CHARS['da'] for c in t),
    'fi': lambda t: ('aa' in t.lower() or 'ii' in t.lower()) and any(c in LANGUAGE_SPECIFIC_CHARS['fi'] for c in t),
    'nl': lambda t: ('ij' in t.lower() or 'sch' in t.lower()) and any(c in LANGUAGE_SPECIFIC_CHARS['nl'] for c in t),
    'ro': lambda t: ('ul' in t.lower() or 'ș' in t or 'ț' in t),
    'is': lambda t: any(c in 'þÞ' for c in t),
    'cs': lambda t: any(char in "řěďťňšžč" for char in t),
    'hu': lambda t: any(char in "őű" for char in t),
    'tr': lambda t: any(char in "ıİğ" for char in t),
    'pl': lambda t: any(char in "łńśźż" for char in t),
    # Added pattern for Indonesian
    'id': lambda t: any(
        word in t.lower() for word in ['yang', 'dan', 'untuk', 'dengan', 'dari', 'universitas', 'indonesia']),
    # Added pattern for Japanese
    'ja': lambda t: any(0x3040 <= ord(c) <= 0x30FF for c in t) or any(0x4E00 <= ord(c) <= 0x9FFF for c in t)
}

# Language patterns for more accurate detection
ACADEMIC_KEYWORDS = {
    'fr': {
        'université': 10, 'école': 8, 'institut': 9, 'laboratoire': 8, 'recherche': 7,
        'france': 6, 'collège': 5, 'école normale': 5, 'centre de recherche': 7,
        'sciences': 4, 'arts': 3, 'médecine': 2, 'faculté': 6, 'département': 5,
        'sorbonne': 8, 'enseignement': 4, 'étude': 3, 'académie': 5
    },
    'es': {
        'universidad': 10, 'instituto': 9, 'investigaciones': 8, 'autónoma': 7,
        'politécnica': 6, 'departamento': 5, 'españa': 5, 'facultad': 4, 'escuela': 3,
        'centro de investigación': 6, 'escuela técnica': 4, 'ciencias': 3, 'tecnología': 6,
        'ingeniería': 5, 'estudios': 5, 'colegio': 4, 'academia': 4
    },
    'id': {
        'universitas': 10, 'indonesia': 9, 'institut': 8, 'penelitian': 7, 'fakultas': 6,
        'pusat': 5, 'teknologi': 5, 'departemen': 4, 'ilmu': 4, 'jakarta': 3, 'bandung': 3,
        'laboratorium': 9, 'teknik': 5, 'studi': 5, 'sekolah': 5, 'perguruan': 4, 'akademi': 4
    },
    'ja': {
        '大学': 10, '研究所': 9, '東京': 9, '日本': 9, '学部': 7, '学院': 6, 'センター': 6,
        '研究センター': 7, '実験室': 9, '学科': 8, '科学': 6, '技術': 6, '工学': 5,
        '研究科': 5, '学校': 5, 'アカデミー': 4
    },
    'nl': {
        'universiteit': 10, 'nederland': 9, 'faculteit': 7, 'instituut': 8, 'centrum': 6,
        'onderzoek': 5, 'technische': 7, 'school': 5, 'academie': 6, 'laboratorium': 9,
        'afdeling': 8, 'wetenschap': 6, 'technologie': 6, 'techniek': 5, 'studie': 5,
        'hogeschool': 4, 'amsterdam': 8, 'leiden': 8, 'utrecht': 8, 'nijmegen': 8
    },
    'en': {
        'university': 10, 'college': 9, 'institute': 8, 'center': 7, 'research': 6,
        'school': 5, 'uk': 6, 'usa': 6, 'canada': 6, 'department': 5, 'faculty': 6,
        'academy': 5, 'institute of technology': 7, 'institute of science': 6,
        'laboratory': 9, 'science': 6, 'technology': 6, 'engineering': 5, 'studies': 5
    },
    'de': {
        'universität': 10, 'hochschule': 9, 'fraunhofer': 8, 'technische': 8,
        'deutschland': 7, 'akademie': 7, 'institut': 6, 'forschungszentrum': 6,
        'schule': 5, 'wissenschaften': 6, 'medizinfakultät': 4, 'berlin': 8, 'münchen': 8
    },
    'it': {
        'università': 10, 'dipartimento': 9, 'italia': 8, 'scuola': 7, 'istituto': 8,
        'centro di ricerca': 6, 'facoltà': 5, 'politecnico': 7, 'accademia': 6,
        'scienze': 4, 'arte': 3, 'roma': 8, 'milano': 8, 'torino': 8
    },
    'pt': {
        'universidade': 10, 'brasil': 9, 'portugal': 9, 'instituto': 8, 'faculdade': 7,
        'departamento': 6, 'escola': 5, 'centro de pesquisa': 6, 'tecnologia': 5, 'ciências': 3,
        'são paulo': 8, 'rio de janeiro': 8, 'lisboa': 8
    }
}


# Unified academic and affiliation keywords (merging academic_common_words and affiliation_hints_weighted)
def _refine_combined_scores(text: str, combined_scores: dict[LangCode: int]) -> dict[LangCode: int]:
    REFINEMENT_WEIGHT = 75

    # First check for highly distinctive characters which are strong indicators
    # Portuguese
    if any(c in 'ãõÃÕ' for c in text):
        combined_scores['pt'] = combined_scores.get('pt', 0) + REFINEMENT_WEIGHT

    # Spanish
    if any(c in 'ñÑ' for c in text):
        combined_scores['es'] = combined_scores.get('es', 0) + REFINEMENT_WEIGHT

    # Nordic languages
    if any(c in 'åÅ' for c in text):
        if any(c in 'æøÆØ' for c in text):
            if 'på' in text.lower():
                combined_scores['no'] = combined_scores.get('no', 0) + REFINEMENT_WEIGHT
            else:
                combined_scores['da'] = combined_scores.get('da', 0) + REFINEMENT_WEIGHT
        else:
            combined_scores['sv'] = combined_scores.get('sv', 0) + REFINEMENT_WEIGHT

    # Hungarian
    if any(c in 'őűŐŰ' for c in text):
        combined_scores['hu'] = combined_scores.get('hu', 0) + REFINEMENT_WEIGHT

    # Czech
    if any(c in 'řěďťňŘĚĎŤŇ' for c in text):
        combined_scores['cs'] = combined_scores.get('cs', 0) + REFINEMENT_WEIGHT

    # Polish
    if any(c in 'łńśźżŁŃŚŹŻ' for c in text):
        combined_scores['pl'] = combined_scores.get('pl', 0) + REFINEMENT_WEIGHT

    # Turkish
    if any(c in 'ıİğĞ' for c in text):
        combined_scores['tr'] = combined_scores.get('tr', 0) + REFINEMENT_WEIGHT

    # Romanian
    if any(c in 'șțȘȚ' for c in text):
        combined_scores['ro'] = combined_scores.get('ro', 0) + REFINEMENT_WEIGHT

    # Icelandic
    if any(c in 'þÞðÐ' for c in text):
        combined_scores['is'] = combined_scores.get('is', 0) + REFINEMENT_WEIGHT

    # German
    if 'ß' in text:
        combined_scores['de'] = combined_scores.get('de', 0) + REFINEMENT_WEIGHT

    # Special case for Indonesian
    if any(word in text.lower() for word in ['indonesia', 'universitas', 'depok', 'jakarta']):
        combined_scores['id'] = combined_scores.get('id', 0) + REFINEMENT_WEIGHT

    # Special case for Dutch
    if 'ij' in text.lower() and any(word in text.lower() for word in ['universiteit', 'nederland', 'amsterdam']):
        combined_scores['nl'] = combined_scores.get('nl', 0) + REFINEMENT_WEIGHT

    return combined_scores


def _score_language_chars(text: str) -> dict[LangCode: float]:
    """
    Score text based on language-specific characters with penalty handling.

    Args:
        text (str): The text to analyze

    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    PENALTY_FACTOR = 2

    lang_chars_count = {}
    for lang, char_set in LANGUAGE_SPECIFIC_CHARS.items():
        count = sum(1 for char in text if char in char_set)
        if lang in INCOMPATIBLE_CHARS:
            penalty = sum(1 for char in text if char in INCOMPATIBLE_CHARS[lang])
            count = max(0, count - penalty * PENALTY_FACTOR)  # Penalty factor can be adjusted
        if count > 0:
            lang_chars_count[lang] = count
    return lang_chars_count


def _score_common_words(text: str, word_dict: dict[str: int]) -> dict[LangCode: float]:
    """
    Calculate weighted scores for common word matches.

    Args:
        text (str): The text to analyze
        word_dict (dict): Dictionary of languages with their common words and weights

    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    # Convert text to lowercase and extract words
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    language_scores = {lang: 0.0 for lang in word_dict}
    word_counts = Counter(words)

    for lang, keywords in word_dict.items():
        for word, weight in keywords.items():
            count = word_counts.get(word, 0)
            if count > 0:
                language_scores[lang] += weight * count

    # Normalize by text length to prevent bias towards longer texts
    text_len = max(1, len(words))
    for lang in language_scores:
        language_scores[lang] = (language_scores[lang] / text_len) * 100

    return {k: v for k, v in language_scores.items() if v > 0}


def _score_academic_keywords(text: str) -> dict[LangCode: float]:
    """
    Calculate weighted scores for academic keyword matches.

    Args:
        text (str): The text to analyze

    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    text_lower = text.lower()
    language_scores = {lang: 0.0 for lang in ACADEMIC_KEYWORDS}

    for lang, keywords in ACADEMIC_KEYWORDS.items():
        for kw, weight in keywords.items():
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                language_scores[lang] += weight

    return language_scores
