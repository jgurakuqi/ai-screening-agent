"""Interfaces shared by all speech-to-text providers."""

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, language: str) -> tuple[str, str]:
        """Transcribe raw audio into text.

        Args:
            audio_bytes: Binary audio payload supplied by the client.
            language: Requested language hint such as ``"es"`` or ``"en"``.

        Returns:
            Tuple of ``(transcript, detected_language)``.
        """

    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Return the language codes the provider can handle directly.

        Returns:
            Supported language codes such as ``["es", "en"]``.
        """
