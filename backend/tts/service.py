"""Text-to-speech service with provider fallback and browser escape hatch."""

from backend.logging_config import logger
from backend.tts.base import TTSProvider
from backend.tts.edge_tts_provider import EdgeTTSProvider


class TTSService:
    """Manages TTS providers with automatic fallback.

    Tries each provider in order. If all fail, it signals the frontend to use
    browser-native speech synthesis as a last resort.
    """

    def __init__(self) -> None:
        self.providers: list[TTSProvider] = [EdgeTTSProvider()]

    def add_provider(self, provider: TTSProvider, priority: int = 0) -> None:
        """Insert a provider into the fallback chain.

        Args:
            provider: Provider instance to register.
            priority: Zero-based insertion index where ``0`` is tried first.
        """
        self.providers.insert(priority, provider)

    async def synthesize(
        self, text: str, language: str
    ) -> tuple[bytes | None, str, bool]:
        """Synthesize speech from text.

        Args:
            text: Text to convert into audio.
            language: Requested language code such as ``"es"`` or ``"en"``.

        Returns:
            Tuple of ``(audio_bytes, content_type, use_browser_fallback)``.
            If all providers fail, ``audio_bytes`` is ``None`` and
            ``use_browser_fallback`` is ``True``.
        """
        for provider in self.providers:
            try:
                audio = await provider.synthesize(text, language)
                return audio, provider.audio_content_type(), False
            except Exception as e:
                logger.warning(
                    "TTS provider {} failed: {}", type(provider).__name__, e
                )
                continue

        logger.warning("All TTS providers failed — signaling browser fallback")
        return None, "", True
