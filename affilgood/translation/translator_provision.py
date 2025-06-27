from affilgood.translation.llm_translator_external import LLMTranslatorExternal
from affilgood.translation.llm_translator_local import LLMTranslatorLocal
from affilgood.translation.model import TranslationConfig


def mk_Translator_from_conf(conf: TranslationConfig):
    if conf.use_external_api:
        return LLMTranslatorExternal(conf=conf)
    else:
        return LLMTranslatorLocal(conf=conf)
