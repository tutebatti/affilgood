from affilgood.language_prediction.model import LangCode


def evaluate_east_asian(text: str) -> LangCode | None:
    # Evaluate East Asian languages with improved differentiation
    jp_score = _score_jp(text)
    kr_score = _score_ko(text)
    cn_score = _score_zh(text)

    # Make decisions based on scores
    if jp_score > 0:
        return LangCode('ja')  # If any clear Japanese indicators, prioritize Japanese

    if kr_score > 0:
        return LangCode('ko')  # If any clear Korean indicators, return Korean

    if cn_score > 0:
        return LangCode('zh')  # If Chinese indicators and no Japanese or Korean, return Chinese

    return None


def _score_jp(text: str) -> float:
    """
    Special detection for Japanese based on specific character ranges.
    """
    HIRAGANA_WEIGHT: float = 2
    KATAKANA_WEIGHT: float = 1.5
    KANJI_WEIGHT: float = 0.5
    JP_PUNCT_WEIGHT: float = 0.5

    # Count characters in Japanese-specific ranges
    hiragana_count = sum(1 for c in text if 0x3040 <= ord(c) <= 0x309F)
    katakana_count = sum(1 for c in text if 0x30A0 <= ord(c) <= 0x30FF)

    # Require hiragana or katakana to identify as Japanese
    if hiragana_count == 0 and katakana_count == 0:
        return 0

    kanji_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)

    # Japanese-specific punctuation
    jp_punct_count = sum(1 for c in text if c in '、。「」『』・')

    # Apply weighted scoring (can be adjusted)
    score = (
            hiragana_count * HIRAGANA_WEIGHT
            + katakana_count * KATAKANA_WEIGHT
            + kanji_count * KANJI_WEIGHT
            + jp_punct_count * JP_PUNCT_WEIGHT
    )

    return score


def _score_ko(text: str) -> float:
    """
    Special detection for Korean based on Hangul presence.
    """
    # Count Hangul characters
    HANGUL_WEIGHT: float = 2
    JAMO_WEIGHT: float = 1.5
    KR_PUNCT_WEIGHT: float = 0.5

    hangul_count = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)

    # Count Hangul Jamo (Korean alphabet components)
    jamo_count = sum(1 for c in text if 0x1100 <= ord(c) <= 0x11FF)

    # If there's no Hangul, it's not Korean
    if hangul_count == 0 and jamo_count == 0:
        return 0

    # Korean-specific punctuation and other Korean-specific ranges
    kr_punct_count = sum(1 for c in text if c in '…·')

    # Apply weighted scoring
    score = (
            hangul_count * HANGUL_WEIGHT
            + jamo_count * JAMO_WEIGHT
            + kr_punct_count * KR_PUNCT_WEIGHT
    )

    return score


def _score_zh(text: str) -> float:
    """
    Special detection for Chinese based on specific character ranges.
    """
    HAN_WEIGHT: float = 1
    ZH_PUNCT_WEIGHT: float = 0.5

    # Check for absence of Japanese-specific characters
    hiragana_count = sum(1 for c in text if 0x3040 <= ord(c) <= 0x309F)
    katakana_count = sum(1 for c in text if 0x30A0 <= ord(c) <= 0x30FF)

    # If there's any kana, it's likely not pure Chinese
    if hiragana_count > 0 or katakana_count > 0:
        return 0

    # Count Han characters (without kana presence)
    han_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)

    # Chinese-specific punctuation
    zh_punct_count = sum(1 for c in text if c in '，。：""''；？！（）')

    # Apply weighted scoring
    score = han_count * HAN_WEIGHT + zh_punct_count * ZH_PUNCT_WEIGHT

    return score
