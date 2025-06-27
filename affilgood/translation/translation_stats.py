from dataclasses import dataclass


@dataclass
class TranslationStats:
    processed: int = 0
    translations_performed: int = 0
    cache_hits: int = 0
    total_processing_time: float = 0.0
    total_translation_time: float = 0.0
    avg_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    avg_translation_time: float = 0.0

    def update(self,
               processed: int,
               performed: int,
               cache_hits: int,
               translation_time: float,
               processing_time: float,
               ) -> None:
        self.processed += processed
        self.translations_performed += performed
        self.cache_hits += cache_hits
        self.total_translation_time += translation_time
        self.total_processing_time += processing_time

    def calculate_additional_metrics(self) -> None:
        if self.processed > 0:
            self.avg_processing_time = self.total_processing_time / self.processed
            self.cache_hit_rate = (self.cache_hits / self.processed) * 100 if self.processed > 0 else 0

        if self.translations_performed > 0:
            self.avg_translation_time = self.total_translation_time / self.translations_performed
