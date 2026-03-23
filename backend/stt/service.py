"""Speech-to-text service with ordered provider fallback."""

from backend.logging_config import logger
from backend.stt.base import STTProvider
from backend.stt.whisper_provider import WhisperSTTProvider


class STTService:
    """Manage STT providers and try them in priority order."""

    def __init__(self) -> None:
        self.providers: list[STTProvider] = [WhisperSTTProvider()]

    def add_provider(self, provider: STTProvider, priority: int = 0) -> None:
        """Insert a provider into the fallback chain.

        Args:
            provider: Provider instance to register.
            priority: Zero-based insertion index where ``0`` is tried first.
        """
        self.providers.insert(priority, provider)

    async def transcribe(self, audio_bytes: bytes, language: str) -> tuple[str, str]:
        """Transcribe audio using the first provider that succeeds.

        Args:
            audio_bytes: Binary audio payload supplied by the API layer.
            language: Requested language hint such as ``"es"`` or ``"en"``.

        Returns:
            Tuple of ``(transcript, detected_language)``.

        Raises:
            RuntimeError: If every configured provider fails.
        """
        for provider in self.providers:
            try:
                return await provider.transcribe(audio_bytes, language)
            except Exception as e:
                logger.warning("STT provider {} failed: {}",
                               type(provider).__name__, e)
                continue
        raise RuntimeError("All STT providers failed")
