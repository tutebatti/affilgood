from affilgood.language_prediction.model import LangCode


def evaluate_non_latin_scripts(text: str) -> LangCode | None:
    # Check for script ranges
    script_scores = _mk_script_scores(text)

    # Handle Cyrillic scripts
    if any(lang in script_scores for lang in ['ru', 'uk', 'bg']):
        if any(char in LANGUAGE_SPECIFIC_CHARS.get('uk', set()) for char in text):
            return LangCode('uk')
        elif any(char in LANGUAGE_SPECIFIC_CHARS.get('bg', set()) for char in text):
            return LangCode('bg')
        elif 'ru' in script_scores:
            return LangCode('ru')

    # Persian vs Arabic
    if 'fa' in script_scores and any(char in LANGUAGE_SPECIFIC_CHARS.get('fa', set()) for char in text):
        return LangCode('fa')
    elif 'ar' in script_scores:
        return LangCode('ar')

    # If we have a match in script ranges, return the one with the highest score
    if script_scores:
        return max(script_scores, key=script_scores.get)

    return None


def _mk_script_scores(text: str) -> dict[LangCode: int]:
    """
    Count characters in each Unicode script range for each language.

    Args:
        text (str): The text to analyze

    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    lang_counts = {}
    for lang, ranges in UNICODE_SCRIPT_RANGES.items():
        count = 0
        for start, end in ranges:
            count += sum(1 for char in text if start <= ord(char) <= end)
        if count > 0:
            lang_counts[lang] = count
    return lang_counts


# Incompatible characters: strong negative signal for a language
INCOMPATIBLE_CHARS = {
    'es': set('àÀèÈùÙìÌâÂêÊîÎôÔûÛëËïÏœŒæÆ'),
    'fr': set('ñÑßẞøØğĞıİčČśŚžŽłŁãÃõÕ'),
    'de': set('ñÑéèêëáàâãäåçîïìíóòôõúùûœŒæÆ'),
    'pt': set('ñÑßẞéèêëîïìíóòôùúûüœŒ'),
    'it': set('ñÑßẞäÄëËïÏöÖüÜæÆøØœŒãÃõÕ'),
    'ca': set('ñÑêÊôÔâÂïÏîÎûÛãÃõÕœŒæÆ'),
    'nl': set('ñÑßẞøØæÆàÀèÈìÌòÒùÙ'),
    'ro': set('ñÑßẞæÆøØñÑìÌùÙ'),
    'sv': set('ñÑßẞæÆœŒ'),
    'da': set('ñÑßẞœŒ'),
    'fi': set('ñÑßẞæÆøØœŒ'),
    'is': set('ñÑßẞæÆœŒ'),
    'pl': set('ñÑßẞæÆøØœŒâÂêÊîÎôÔûÛ'),
    'cs': set('ñÑßẞæÆøØœŒ'),
    'hu': set('ñÑßẞæÆøØœŒ'),
    'tr': set('ñÑßẞæÆøØœŒ'),
    'no': set('ñÑßẞœŒ'),
    'id': set('äÄëËïÏöÖüÜæÆøØñÑßẞœŒ'),  # Indonesian
}

# Language-specific characters: strong positive signal for a language
LANGUAGE_SPECIFIC_CHARS = {
    'es': set('áéíñóúüÁÉÍÑÓÚÜ¿¡'),
    'fr': set('àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ'),
    'de': set('äöüßÄÖÜ'),
    'pt': set('áàâãçéêíóôõúÁÀÂÃÇÉÊÍÓÔÕÚ'),
    'it': set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'),
    'ca': set('àçèéíïòóúüÀÇÈÉÍÏÒÓÚÜ·'),
    'nl': set('áéëíóúüïèêÁÉËÍÓÚÜÏÈÊ'),
    'ro': set('ăâîșțĂÂÎȘȚ'),
    'sv': set('åäöÅÄÖ'),
    'da': set('åæøÅÆØ'),
    'fi': set('äöÄÖ'),
    'is': set('áðéíóúýþæöÁÐÉÍÓÚÝÞÆÖ'),
    'pl': set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'),
    'cs': set('čďěňřšťůžČĎĚŇŘŠŤŮŽ'),
    'hu': set('őűŐŰ'),
    'tr': set('çğıİöşüÇĞÖŞÜ'),
    'uk': set('їієґЇІЄҐ'),
    'bg': set('ъьЪЬ'),
    'fa': set('پچژگ'),
    'no': set('åæøÅÆØ'),
    'id': set(''),  # Indonesian generally uses ASCII but with specific patterns
}

# Unicode script ranges for non-Latin scripts
UNICODE_SCRIPT_RANGES = {
    'ru': [(0x0400, 0x04FF)],  # Russian - Basic Cyrillic
    'uk': [(0x0400, 0x04FF)],  # Ukrainian
    'bg': [(0x0400, 0x04FF)],  # Bulgarian
    'el': [(0x0370, 0x03FF)],  # Greek
    'pl': [(0x0100, 0x024F)],  # Polish
    'cs': [(0x0100, 0x024F)],  # Czech
    'hu': [(0x0100, 0x024F)],  # Hungarian
    'tr': [(0x0100, 0x024F)],  # Turkish
    'ar': [(0x0600, 0x06FF), (0x0750, 0x077F)],  # Arabic
    'he': [(0x0590, 0x05FF)],  # Hebrew
    'fa': [(0x0600, 0x06FF), (0xFB50, 0xFDFF), (0x0750, 0x077F)],  # Persian (Farsi)
    'zh': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],  # Chinese
    'ja': [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF00, 0xFFEF), (0x4E00, 0x9FFF)],  # Japanese
    'ko': [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],  # Korean
    'hi': [(0x0900, 0x097F)],  # Hindi (Devanagari)
    'bn': [(0x0980, 0x09FF)],  # Bengali
    'ta': [(0x0B80, 0x0BFF)],  # Tamil
    'te': [(0x0C00, 0x0C7F)],  # Telugu
    'kn': [(0x0C80, 0x0CFF)],  # Kannada
    'ml': [(0x0D00, 0x0D7F)],  # Malayalam
    'th': [(0x0E00, 0x0E7F)],  # Thai
    'km': [(0x1780, 0x17FF)],  # Khmer
    'my': [(0x1000, 0x109F)],  # Burmese (Myanmar)
    'lo': [(0x0E80, 0x0EFF)],  # Lao
    'am': [(0x1200, 0x137F)],  # Amharic (Ethiopic)
    'ka': [(0x10A0, 0x10FF)],  # Georgian
    'hy': [(0x0530, 0x058F)],  # Armenian
}
